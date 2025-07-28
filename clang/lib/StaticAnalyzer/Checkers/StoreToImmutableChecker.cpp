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

private:
  bool isConstVariable(const MemRegion *MR, CheckerContext &C) const;
  bool isConstQualifiedType(const MemRegion *MR, CheckerContext &C) const;
};
} // end anonymous namespace

bool StoreToImmutableChecker::isConstVariable(const MemRegion *MR,
                                              CheckerContext &C) const {
  // Check if the region is in the global immutable space
  const MemSpaceRegion *MS = MR->getMemorySpace(C.getState());
  if (isa<GlobalImmutableSpaceRegion>(MS))
    return true;

  // Check if this is a VarRegion with a const-qualified type
  if (const VarRegion *VR = dyn_cast<VarRegion>(MR)) {
    const VarDecl *VD = VR->getDecl();
    if (VD && VD->getType().isConstQualified())
      return true;
  }

  // Check if this is a FieldRegion with a const-qualified type
  if (const FieldRegion *FR = dyn_cast<FieldRegion>(MR)) {
    const FieldDecl *FD = FR->getDecl();
    if (FD && FD->getType().isConstQualified())
      return true;
  }

  // Check if this is a SymbolicRegion with a const-qualified pointee type
  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(MR)) {
    QualType PointeeType = SR->getPointeeStaticType();
    if (PointeeType.isConstQualified())
      return true;
  }

  // Check if this is an ElementRegion accessing a const array
  if (const ElementRegion *ER = dyn_cast<ElementRegion>(MR)) {
    return isConstQualifiedType(ER->getSuperRegion(), C);
  }

  return false;
}

bool StoreToImmutableChecker::isConstQualifiedType(const MemRegion *MR,
                                                   CheckerContext &C) const {
  // Check if the region has a const-qualified type
  if (const TypedValueRegion *TVR = dyn_cast<TypedValueRegion>(MR)) {
    QualType Ty = TVR->getValueType();
    return Ty.isConstQualified();
  }
  return false;
}

void StoreToImmutableChecker::checkBind(SVal Loc, SVal Val, const Stmt *S,
                                        CheckerContext &C) const {
  // We are only interested in stores to memory regions
  const MemRegion *MR = Loc.getAsRegion();
  if (!MR)
    return;

  // Skip variable declarations and initializations - we only want to catch
  // actual writes
  if (isa<DeclStmt, DeclRefExpr>(S))
    return;

  // Check if the region corresponds to a const variable
  if (!isConstVariable(MR, C))
    return;

  // Generate the bug report
  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;

  constexpr llvm::StringLiteral Msg =
      "Writing to immutable memory is undefined behavior. "
      "This memory region is marked as immutable and should not be modified.";

  auto R = std::make_unique<PathSensitiveBugReport>(BT, Msg, N);
  R->addRange(S->getSourceRange());

  // If the location that is being written to has a declaration, place a note.
  if (const DeclRegion *DR = dyn_cast<DeclRegion>(MR)) {
    R->addNote(
        "Memory region is in immutable space",
        PathDiagnosticLocation::create(DR->getDecl(), C.getSourceManager()));
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
