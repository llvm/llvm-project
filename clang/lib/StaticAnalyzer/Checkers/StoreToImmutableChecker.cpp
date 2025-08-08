//=== StoreToImmutableChecker.cpp - Store to immutable memory ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines StoreToImmutableChecker, a checker that detects writes
// to immutable memory regions. This implements part of SEI CERT Rule ENV30-C.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"

using namespace clang;
using namespace ento;

namespace {
class StoreToImmutableChecker : public Checker<check::Bind> {
  const BugType BT{this, "Write to immutable memory", "CERT Environment (ENV)"};

public:
  void checkBind(SVal Loc, SVal Val, const Stmt *S, CheckerContext &C) const;
};
} // end anonymous namespace

static bool isInitializationContext(const Stmt *S, CheckerContext &C) {
  // Check if this is a DeclStmt (variable declaration)
  if (isa<DeclStmt>(S))
    return true;

  // This part is specific for initialization of const lambdas pre-C++17.
  // Lets look at the AST of the statement:
  // ```
  // const auto lambda = [](){};
  // ```
  //
  // The relevant part of the AST for this case prior to C++17 is:
  // ...
  // `-DeclStmt
  //   `-VarDecl
  //     `-ExprWithCleanups
  //       `-CXXConstructExpr
  // ...
  // In C++17 and later, the AST is different:
  // ...
  // `-DeclStmt
  //   `-VarDecl
  //     `-ImplicitCastExpr
  //       `-LambdaExpr
  //         |-CXXRecordDecl
  //         `-CXXConstructExpr
  // ...
  // And even beside this, the statement `S` that is given to the checkBind
  // callback is the VarDecl in C++17 and later, and the CXXConstructExpr in
  // C++14 and before. So in order to support the C++14 we need the following
  // ugly hack to detect whether this construction is used to initialize a
  // variable.
  //
  // FIXME: This should be eliminated by improving the API of checkBind to
  // ensure that it consistently passes the `VarDecl` (instead of the
  // `CXXConstructExpr`) when the constructor call denotes the initialization
  // of a variable with a lambda, or maybe less preferably, try the more
  // invasive approach of passing the information forward to the checkers
  // whether the current bind is an initialization or an assignment.
  const auto *ConstructExp = dyn_cast<CXXConstructExpr>(S);
  return ConstructExp && ConstructExp->isElidable();
}

static bool isEffectivelyConstRegion(const MemRegion *MR, CheckerContext &C) {
  if (isa<GlobalImmutableSpaceRegion>(MR))
    return true;

  // Check if this is a TypedRegion with a const-qualified type
  if (const auto *TR = dyn_cast<TypedRegion>(MR)) {
    QualType LocationType = TR->getDesugaredLocationType(C.getASTContext());
    if (LocationType->isPointerOrReferenceType())
      LocationType = LocationType->getPointeeType();
    if (LocationType.isConstQualified())
      return true;
  }

  // Check if this is a SymbolicRegion with a const-qualified pointee type
  if (const auto *SR = dyn_cast<SymbolicRegion>(MR)) {
    QualType PointeeType = SR->getPointeeStaticType();
    if (PointeeType.isConstQualified())
      return true;
  }

  // NOTE: The above branches do not cover AllocaRegion. We do not need to check
  // AllocaRegion, as it models untyped memory, that is allocated on the stack.

  return false;
}

static const MemRegion *getInnermostConstRegion(const MemRegion *MR,
                                                CheckerContext &C) {
  while (true) {
    if (isEffectivelyConstRegion(MR, C))
      return MR;
    if (auto *SR = dyn_cast<SubRegion>(MR))
      MR = SR->getSuperRegion();
    else
      return nullptr;
  }
}

static const DeclRegion *
getInnermostEnclosingConstDeclRegion(const MemRegion *MR, CheckerContext &C) {
  while (true) {
    if (const auto *DR = dyn_cast<DeclRegion>(MR)) {
      const ValueDecl *D = DR->getDecl();
      QualType DeclaredType = D->getType();
      if (DeclaredType.isConstQualified())
        return DR;
    }
    if (auto *SR = dyn_cast<SubRegion>(MR))
      MR = SR->getSuperRegion();
    else
      return nullptr;
  }
}

void StoreToImmutableChecker::checkBind(SVal Loc, SVal Val, const Stmt *S,
                                        CheckerContext &C) const {
  // We are only interested in stores to memory regions
  const MemRegion *MR = Loc.getAsRegion();
  if (!MR)
    return;

  // Skip variable declarations and initializations - we only want to catch
  // actual writes
  // FIXME: If the API of checkBind would allow to distinguish between
  // initialization and assignment, we could use that instead.
  if (isInitializationContext(S, C))
    return;

  // Check if the region is in the global immutable space
  const MemSpaceRegion *MS = MR->getMemorySpace(C.getState());
  const bool IsGlobalImmutableSpace = isa<GlobalImmutableSpaceRegion>(MS);
  // Check if the region corresponds to a const variable
  const MemRegion *InnermostConstRegion = getInnermostConstRegion(MR, C);
  if (!IsGlobalImmutableSpace && !InnermostConstRegion)
    return;

  SmallString<64> WarningMessage{"Trying to write to immutable memory"};
  if (IsGlobalImmutableSpace)
    WarningMessage += " in global read-only storage";

  // Generate the bug report
  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;

  auto R = std::make_unique<PathSensitiveBugReport>(BT, WarningMessage, N);
  R->addRange(S->getSourceRange());

  // Generate a note if the location that is being written to has a
  // declaration or if it is a subregion of a const region with a declaration.
  const DeclRegion *DR =
      getInnermostEnclosingConstDeclRegion(InnermostConstRegion, C);
  if (DR) {
    const char *NoteMessage =
        (DR != MR) ? "Enclosing memory region is declared as immutable here"
                   : "Memory region is declared as immutable here";
    R->addNote(NoteMessage, PathDiagnosticLocation::create(
                                DR->getDecl(), C.getSourceManager()));
  }

  // For this checker, we are only interested in the value being written, no
  // need to mark the value being assigned interesting.

  C.emitReport(std::move(R));
}

void ento::registerStoreToImmutableChecker(CheckerManager &mgr) {
  mgr.registerChecker<StoreToImmutableChecker>();
}

bool ento::shouldRegisterStoreToImmutableChecker(const CheckerManager &mgr) {
  return true;
}
