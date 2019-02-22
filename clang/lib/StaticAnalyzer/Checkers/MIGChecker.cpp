//== MIGChecker.cpp - MIG calling convention checker ------------*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MIGChecker, a Mach Interface Generator calling convention
// checker. Namely, in MIG callback implementation the following rules apply:
// - When a server routine returns KERN_SUCCESS, it must take ownership of
//   resources (and eventually release them).
// - Additionally, when returning KERN_SUCCESS, all out-parameters must be
//   initialized.
// - When it returns anything except KERN_SUCCESS it must not take ownership,
//   because the message and its descriptors will be destroyed by the server
//   function.
// For now we only check the last rule, as its violations lead to dangerous
// use-after-free exploits.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/AnyCall.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class MIGChecker : public Checker<check::PostCall, check::PreStmt<ReturnStmt>> {
  BugType BT{this, "Use-after-free (MIG calling convention violation)",
             categories::MemoryError};

  CallDescription vm_deallocate { "vm_deallocate", 3 };

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const;

  class Visitor : public BugReporterVisitor {
  public:
    void Profile(llvm::FoldingSetNodeID &ID) const {
      static int X = 0;
      ID.AddPointer(&X);
    }

    std::shared_ptr<PathDiagnosticPiece> VisitNode(const ExplodedNode *N,
        BugReporterContext &BRC, BugReport &R);
  };
};
} // end anonymous namespace

// FIXME: It's a 'const ParmVarDecl *' but there's no ready-made GDM traits
// specialization for this sort of types.
REGISTER_TRAIT_WITH_PROGRAMSTATE(ReleasedParameter, const void *)

std::shared_ptr<PathDiagnosticPiece>
MIGChecker::Visitor::VisitNode(const ExplodedNode *N, BugReporterContext &BRC,
                               BugReport &R) {
  const auto *NewPVD = static_cast<const ParmVarDecl *>(
      N->getState()->get<ReleasedParameter>());
  const auto *OldPVD = static_cast<const ParmVarDecl *>(
      N->getFirstPred()->getState()->get<ReleasedParameter>());
  if (OldPVD == NewPVD)
    return nullptr;

  assert(NewPVD && "What is deallocated cannot be un-deallocated!");
  SmallString<64> Str;
  llvm::raw_svector_ostream OS(Str);
  OS << "Value passed through parameter '" << NewPVD->getName()
     << "' is deallocated";

  PathDiagnosticLocation Loc =
      PathDiagnosticLocation::create(N->getLocation(), BRC.getSourceManager());
  return std::make_shared<PathDiagnosticEventPiece>(Loc, OS.str());
}

static const ParmVarDecl *getOriginParam(SVal V, CheckerContext &C) {
  SymbolRef Sym = V.getAsSymbol();
  if (!Sym)
    return nullptr;

  const auto *VR = dyn_cast_or_null<VarRegion>(Sym->getOriginRegion());
  if (VR && VR->hasStackParametersStorage() &&
         VR->getStackFrame()->inTopFrame())
    return cast<ParmVarDecl>(VR->getDecl());

  return nullptr;
}

static bool isInMIGCall(CheckerContext &C) {
  const LocationContext *LC = C.getLocationContext();
  const StackFrameContext *SFC;
  // Find the top frame.
  while (LC) {
    SFC = LC->getStackFrame();
    LC = SFC->getParent();
  }

  const Decl *D = SFC->getDecl();

  if (Optional<AnyCall> AC = AnyCall::forDecl(D)) {
    // Even though there's a Sema warning when the return type of an annotated
    // function is not a kern_return_t, this warning isn't an error, so we need
    // an extra sanity check here.
    // FIXME: AnyCall doesn't support blocks yet, so they remain unchecked
    // for now.
    if (!AC->getReturnType(C.getASTContext())
             .getCanonicalType()->isSignedIntegerType())
      return false;
  }

  if (D->hasAttr<MIGServerRoutineAttr>())
    return true;

  return false;
}

void MIGChecker::checkPostCall(const CallEvent &Call, CheckerContext &C) const {
  if (!isInMIGCall(C))
    return;

  if (!Call.isGlobalCFunction())
    return;

  if (!Call.isCalled(vm_deallocate))
    return;

  // TODO: Unhardcode "1".
  SVal Arg = Call.getArgSVal(1);
  const ParmVarDecl *PVD = getOriginParam(Arg, C);
  if (!PVD)
    return;

  C.addTransition(C.getState()->set<ReleasedParameter>(PVD));
}

void MIGChecker::checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const {
  // It is very unlikely that a MIG callback will be called from anywhere
  // within the project under analysis and the caller isn't itself a routine
  // that follows the MIG calling convention. Therefore we're safe to believe
  // that it's always the top frame that is of interest. There's a slight chance
  // that the user would want to enforce the MIG calling convention upon
  // a random routine in the middle of nowhere, but given that the convention is
  // fairly weird and hard to follow in the first place, there's relatively
  // little motivation to spread it this way.
  if (!C.inTopFrame())
    return;

  if (!isInMIGCall(C))
    return;

  // We know that the function is non-void, but what if the return statement
  // is not there in the code? It's not a compile error, we should not crash.
  if (!RS)
    return;

  ProgramStateRef State = C.getState();
  if (!State->get<ReleasedParameter>())
    return;

  SVal V = C.getSVal(RS);
  if (!State->isNonNull(V).isConstrainedTrue())
    return;

  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;

  auto R = llvm::make_unique<BugReport>(
      BT,
      "MIG callback fails with error after deallocating argument value. "
      "This is a use-after-free vulnerability because the caller will try to "
      "deallocate it again",
      N);

  R->addRange(RS->getSourceRange());
  bugreporter::trackExpressionValue(N, RS->getRetValue(), *R, false);
  R->addVisitor(llvm::make_unique<Visitor>());
  C.emitReport(std::move(R));
}

void ento::registerMIGChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<MIGChecker>();
}

bool ento::shouldRegisterMIGChecker(const LangOptions &LO) {
  return true;
}
