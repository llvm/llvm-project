//===-- ChrootChecker.cpp - chroot usage checks ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines chroot checker, which checks improper use of chroot. This
//  is described by the SEI Cert C rule POS05-C. The checker is a warning not
//  a hard failure since it only checks for a recommended rule.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"

using namespace clang;
using namespace ento;

// enum value that represent the jail state
enum ChrootKind { NO_CHROOT, ROOT_CHANGED, ROOT_CHANGE_FAILED, JAIL_ENTERED };

// Track chroot state changes for success, failure, state change
// and "jail"
REGISTER_TRAIT_WITH_PROGRAMSTATE(ChrootState, ChrootKind)
namespace {

// This checker checks improper use of chroot.
// The state transitions
//
//                          -> ROOT_CHANGE_FAILED
//                          |
// NO_CHROOT ---chroot(path)--> ROOT_CHANGED ---chdir(/) --> JAIL_ENTERED
//                                  |                               |
//         ROOT_CHANGED<--chdir(..)--      JAIL_ENTERED<--chdir(..)--
//                                  |                               |
//                      bug<--foo()--          JAIL_ENTERED<--foo()--
//
class ChrootChecker : public Checker<eval::Call, check::PreCall> {
  // This bug refers to possibly break out of a chroot() jail.
  const BugType BT_BreakJail{this, "Break out of jail"};

  const CallDescription Chroot{CDM::CLibrary, {"chroot"}, 1},
      Chdir{CDM::CLibrary, {"chdir"}, 1};

public:
  ChrootChecker() {}

  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

private:
  void evalChroot(const CallEvent &Call, CheckerContext &C) const;
  void evalChdir(const CallEvent &Call, CheckerContext &C) const;
};

bool ChrootChecker::evalCall(const CallEvent &Call, CheckerContext &C) const {
  if (Chroot.matches(Call)) {
    evalChroot(Call, C);
    return true;
  }
  if (Chdir.matches(Call)) {
    evalChdir(Call, C);
    return true;
  }

  return false;
}

void ChrootChecker::evalChroot(const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();
  BasicValueFactory &BVF = SVB.getBasicValueFactory();

  const QualType IntTy = C.getASTContext().IntTy;

  // Using CallDescriptions to match on CallExpr, so no need
  // to do null checks.
  const Expr *ChrootCE = Call.getOriginExpr();
  const auto *CE = cast<CallExpr>(ChrootCE);

  const LocationContext *LCtx = C.getLocationContext();

  const llvm::APSInt &Zero = BVF.getValue(0, IntTy);
  const llvm::APSInt &Minus1 = BVF.getValue(-1, IntTy);

  // Setup path where chroot fails, returning -1
  ProgramStateRef StateChrootFailed =
      State->BindExpr(CE, LCtx, nonloc::ConcreteInt{Minus1});
  if (StateChrootFailed) {
    StateChrootFailed = StateChrootFailed->set<ChrootState>(ROOT_CHANGE_FAILED);
    C.addTransition(StateChrootFailed);
  }

  // Setup path where chroot succeeds, returning 0
  ProgramStateRef StateChrootSuccess =
      State->BindExpr(CE, LCtx, nonloc::ConcreteInt{Zero});
  if (StateChrootSuccess) {
    StateChrootSuccess = StateChrootSuccess->set<ChrootState>(ROOT_CHANGED);
    C.addTransition(StateChrootSuccess);
  }
}

void ChrootChecker::evalChdir(const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  // If there are no jail state, just return.
  if (State->get<ChrootState>() == NO_CHROOT)
    return;

  // After chdir("/"), enter the jail, set the enum value JAIL_ENTERED.
  SVal ArgVal = Call.getArgSVal(0);

  if (const MemRegion *R = ArgVal.getAsRegion()) {
    R = R->StripCasts();
    if (const auto *StrRegion = dyn_cast<StringRegion>(R)) {
      if (StrRegion->getStringLiteral()->getString() == "/") {
        C.addTransition(State->set<ChrootState>(JAIL_ENTERED));
      }
    }
  }
}

class ChrootInvocationVisitor final : public BugReporterVisitor {
public:
  explicit ChrootInvocationVisitor(const CallDescription &Chroot)
      : Chroot{Chroot} {}

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override {
    if (Satisfied)
      return nullptr;

    auto StmtP = N->getLocation().getAs<StmtPoint>();
    if (!StmtP)
      return nullptr;

    const CallExpr *Call = StmtP->getStmtAs<CallExpr>();
    if (!Call)
      return nullptr;

    if (!matchesAnyAsWritten(*Call, Chroot))
      return nullptr;

    Satisfied = true;
    PathDiagnosticLocation Pos(Call, BRC.getSourceManager(),
                               N->getLocationContext());
    return std::make_shared<PathDiagnosticEventPiece>(Pos, "chroot called here",
                                                      /*addPosRange=*/true);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static bool Tag;
    ID.AddPointer(&Tag);
  }

private:
  const CallDescription &Chroot;
  bool Satisfied = false;
};

// Check the jail state before any function call except chroot and chdir().
void ChrootChecker::checkPreCall(const CallEvent &Call,
                                 CheckerContext &C) const {
  // Ignore chroot and chdir.
  if (matchesAny(Call, Chroot, Chdir))
    return;

  // If jail state is not ROOT_CHANGED just return.
  if (C.getState()->get<ChrootState>() != ROOT_CHANGED)
    return;

  // Generate bug report.
  ExplodedNode *Err =
      C.generateNonFatalErrorNode(C.getState(), C.getPredecessor());
  if (!Err)
    return;

  std::unique_ptr<PathSensitiveBugReport> R =
      std::make_unique<PathSensitiveBugReport>(
          BT_BreakJail, R"(No call of chdir("/") immediately after chroot)",
          Err);
  R->addVisitor<ChrootInvocationVisitor>(Chroot);
  C.emitReport(std::move(R));
}

} // namespace

void ento::registerChrootChecker(CheckerManager &mgr) {
  mgr.registerChecker<ChrootChecker>();
}

bool ento::shouldRegisterChrootChecker(const CheckerManager &mgr) {
  return true;
}
