//===-- SetgidSetuidOrderChecker.cpp - check privilege revocation calls ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a checker to detect possible reversed order of privilege
//  revocations when 'setgid' and 'setuid' is used.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"

using namespace clang;
using namespace ento;

namespace {

enum SetPrivilegeFunctionKind { Irrelevant, Setuid, Setgid };

class SetgidSetuidOrderChecker
    : public Checker<check::PostCall, check::DeadSymbols, eval::Assume> {
  const BugType BT{this, "Possible wrong order of privilege revocation"};

  const CallDescription SetuidDesc{CDM::CLibrary, {"setuid"}, 1};
  const CallDescription SetgidDesc{CDM::CLibrary, {"setgid"}, 1};

  const CallDescription GetuidDesc{CDM::CLibrary, {"getuid"}, 0};
  const CallDescription GetgidDesc{CDM::CLibrary, {"getgid"}, 0};

  CallDescriptionSet const OtherSetPrivilegeDesc{
      {CDM::CLibrary, {"seteuid"}, 1},   {CDM::CLibrary, {"setegid"}, 1},
      {CDM::CLibrary, {"setreuid"}, 2},  {CDM::CLibrary, {"setregid"}, 2},
      {CDM::CLibrary, {"setresuid"}, 3}, {CDM::CLibrary, {"setresgid"}, 3}};

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  ProgramStateRef evalAssume(ProgramStateRef State, SVal Cond,
                             bool Assumption) const;

private:
  ProgramStateRef processSetuid(ProgramStateRef State, const CallEvent &Call,
                                CheckerContext &C) const;
  ProgramStateRef processSetgid(ProgramStateRef State, const CallEvent &Call,
                                CheckerContext &C) const;
  ProgramStateRef processOther(ProgramStateRef State, const CallEvent &Call,
                               CheckerContext &C) const;
  /// Check if a function like \c getuid or \c getgid is called directly from
  /// the first argument of function called from \a Call.
  bool isFunctionCalledInArg(const CallDescription &Desc,
                             const CallEvent &Call) const;
  void emitReport(ProgramStateRef State, CheckerContext &C) const;
};

} // end anonymous namespace

/// Store if there was a call to 'setuid(getuid())' or 'setgid(getgid())' not
/// followed by other different privilege-change functions.
/// If the value \c Setuid is stored and a 'setgid(getgid())' call is found we
/// have found the bug to be reported. Value \c Setgid is used too to prevent
/// warnings at a setgid-setuid-setgid sequence.
REGISTER_TRAIT_WITH_PROGRAMSTATE(LastSetPrivilegeCall, SetPrivilegeFunctionKind)
/// Store the symbol value of the last 'setuid(getuid())' call. This is used to
/// detect if the result is compared to -1 and avoid warnings on that branch
/// (which is the failure branch of the call).
REGISTER_TRAIT_WITH_PROGRAMSTATE(LastSetuidCallSVal, SymbolRef)

void SetgidSetuidOrderChecker::checkPostCall(const CallEvent &Call,
                                             CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  if (SetuidDesc.matches(Call)) {
    State = processSetuid(State, Call, C);
  } else if (SetgidDesc.matches(Call)) {
    State = processSetgid(State, Call, C);
  } else if (OtherSetPrivilegeDesc.contains(Call)) {
    State = processOther(State, Call, C);
  }
  if (State)
    C.addTransition(State);
}

void SetgidSetuidOrderChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                                CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  SymbolRef LastSetuidSym = State->get<LastSetuidCallSVal>();
  if (!LastSetuidSym)
    return;

  if (!SymReaper.isDead(LastSetuidSym))
    return;

  State = State->set<LastSetuidCallSVal>(SymbolRef{});
  C.addTransition(State);
}

ProgramStateRef SetgidSetuidOrderChecker::evalAssume(ProgramStateRef State,
                                                     SVal Cond,
                                                     bool Assumption) const {
  SValBuilder &SVB = State->getStateManager().getSValBuilder();
  SymbolRef LastSetuidSym = State->get<LastSetuidCallSVal>();
  if (!LastSetuidSym)
    return State;

  // Check if the most recent call to 'setuid(getuid())' is assumed to be != 0.
  // It should be only -1 at failure, but we want to accept a "!= 0" check too.
  // (But now an invalid failure check like "!= 1" will be recognized as correct
  // too. The "invalid failure check" is a different bug that is not the scope
  // of this checker.)
  auto FailComparison =
      SVB.evalBinOpNN(State, BO_NE, nonloc::SymbolVal(LastSetuidSym),
                      SVB.makeIntVal(0, /*isUnsigned=*/false),
                      SVB.getConditionType())
          .getAs<DefinedOrUnknownSVal>();
  if (!FailComparison)
    return State;
  if (auto IsFailBranch = State->assume(*FailComparison);
      IsFailBranch.first && !IsFailBranch.second) {
    // This is the 'setuid(getuid())' != 0 case.
    // On this branch we do not want to emit warning.
    State = State->set<LastSetuidCallSVal>(SymbolRef{});
    State = State->set<LastSetPrivilegeCall>(Irrelevant);
  }
  return State;
}

ProgramStateRef SetgidSetuidOrderChecker::processSetuid(
    ProgramStateRef State, const CallEvent &Call, CheckerContext &C) const {
  bool IsSetuidWithGetuid = isFunctionCalledInArg(GetuidDesc, Call);
  if (State->get<LastSetPrivilegeCall>() != Setgid && IsSetuidWithGetuid) {
    State = State->set<LastSetuidCallSVal>(Call.getReturnValue().getAsSymbol());
    return State->set<LastSetPrivilegeCall>(Setuid);
  }
  State = State->set<LastSetuidCallSVal>(SymbolRef{});
  return State->set<LastSetPrivilegeCall>(Irrelevant);
}

ProgramStateRef SetgidSetuidOrderChecker::processSetgid(
    ProgramStateRef State, const CallEvent &Call, CheckerContext &C) const {
  bool IsSetgidWithGetgid = isFunctionCalledInArg(GetgidDesc, Call);
  State = State->set<LastSetuidCallSVal>(SymbolRef{});
  if (State->get<LastSetPrivilegeCall>() == Setuid) {
    if (IsSetgidWithGetgid) {
      State = State->set<LastSetPrivilegeCall>(Irrelevant);
      emitReport(State, C);
      // return nullptr to prevent adding transition with the returned state
      return nullptr;
    }
    return State->set<LastSetPrivilegeCall>(Irrelevant);
  }
  return State->set<LastSetPrivilegeCall>(IsSetgidWithGetgid ? Setgid
                                                             : Irrelevant);
}

ProgramStateRef SetgidSetuidOrderChecker::processOther(
    ProgramStateRef State, const CallEvent &Call, CheckerContext &C) const {
  State = State->set<LastSetuidCallSVal>(SymbolRef{});
  return State->set<LastSetPrivilegeCall>(Irrelevant);
}

bool SetgidSetuidOrderChecker::isFunctionCalledInArg(
    const CallDescription &Desc, const CallEvent &Call) const {
  if (const auto *CallInArg0 =
          dyn_cast<CallExpr>(Call.getArgExpr(0)->IgnoreParenImpCasts()))
    return Desc.matchesAsWritten(*CallInArg0);
  return false;
}

void SetgidSetuidOrderChecker::emitReport(ProgramStateRef State,
                                          CheckerContext &C) const {
  if (ExplodedNode *N = C.generateNonFatalErrorNode(State)) {
    llvm::StringLiteral Msg =
        "A 'setgid(getgid())' call following a 'setuid(getuid())' "
        "call is likely to fail; probably the order of these "
        "statements is wrong";
    C.emitReport(std::make_unique<PathSensitiveBugReport>(BT, Msg, N));
  }
}

void ento::registerSetgidSetuidOrderChecker(CheckerManager &mgr) {
  mgr.registerChecker<SetgidSetuidOrderChecker>();
}

bool ento::shouldRegisterSetgidSetuidOrderChecker(const CheckerManager &mgr) {
  return true;
}
