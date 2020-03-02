//===-- StreamChecker.cpp -----------------------------------------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines checkers that model and check stream handling functions.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include <functional>

using namespace clang;
using namespace ento;
using namespace std::placeholders;

namespace {

struct StreamState {
  enum Kind { Opened, Closed, OpenFailed, Escaped } K;

  StreamState(Kind k) : K(k) {}

  bool isOpened() const { return K == Opened; }
  bool isClosed() const { return K == Closed; }
  //bool isOpenFailed() const { return K == OpenFailed; }
  //bool isEscaped() const { return K == Escaped; }

  bool operator==(const StreamState &X) const { return K == X.K; }

  static StreamState getOpened() { return StreamState(Opened); }
  static StreamState getClosed() { return StreamState(Closed); }
  static StreamState getOpenFailed() { return StreamState(OpenFailed); }
  static StreamState getEscaped() { return StreamState(Escaped); }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
  }
};

class StreamChecker;

using FnCheck = std::function<void(const StreamChecker *, const CallEvent &,
                                   CheckerContext &)>;

struct FnDescription {
  FnCheck EvalFn;
};

class StreamChecker : public Checker<eval::Call,
                                     check::DeadSymbols > {
  mutable std::unique_ptr<BuiltinBug> BT_nullfp, BT_illegalwhence,
      BT_doubleclose, BT_ResourceLeak;

public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;

private:

  CallDescriptionMap<FnDescription> FnDescriptions = {
      {{"fopen"}, {&StreamChecker::evalFopen}},
      {{"freopen", 3}, {&StreamChecker::evalFreopen}},
      {{"tmpfile"}, {&StreamChecker::evalFopen}},
      {{"fclose", 1}, {&StreamChecker::evalFclose}},
      {{"fread", 4},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 3)}},
      {{"fwrite", 4},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 3)}},
      {{"fseek", 3}, {&StreamChecker::evalFseek}},
      {{"ftell", 1},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 0)}},
      {{"rewind", 1},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 0)}},
      {{"fgetpos", 2},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 0)}},
      {{"fsetpos", 2},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 0)}},
      {{"clearerr", 1},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 0)}},
      {{"feof", 1},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 0)}},
      {{"ferror", 1},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 0)}},
      {{"fileno", 1},
       {std::bind(&StreamChecker::checkArgNullStream, _1, _2, _3, 0)}},
  };

  void evalFopen(const CallEvent &Call, CheckerContext &C) const;
  void evalFreopen(const CallEvent &Call, CheckerContext &C) const;
  void evalFclose(const CallEvent &Call, CheckerContext &C) const;
  void evalFseek(const CallEvent &Call, CheckerContext &C) const;
  void checkArgNullStream(const CallEvent &Call, CheckerContext &C,
                          unsigned ArgI) const;

  ProgramStateRef checkNullStream(SVal SV, CheckerContext &C,
                                  ProgramStateRef State) const;
  ProgramStateRef checkFseekWhence(SVal SV, CheckerContext &C,
                                   ProgramStateRef State) const;
  ProgramStateRef checkDoubleClose(const CallEvent &Call, CheckerContext &C,
                                   ProgramStateRef State) const;
};

} // end anonymous namespace

REGISTER_MAP_WITH_PROGRAMSTATE(StreamMap, SymbolRef, StreamState)


bool StreamChecker::evalCall(const CallEvent &Call, CheckerContext &C) const {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD || FD->getKind() != Decl::Function)
    return false;

  // Recognize "global C functions" with only integral or pointer arguments
  // (and matching name) as stream functions.
  if (!Call.isGlobalCFunction())
    return false;
  for (auto P : Call.parameters()) {
    QualType T = P->getType();
    if (!T->isIntegralOrEnumerationType() && !T->isPointerType())
      return false;
  }

  const FnDescription *Description = FnDescriptions.lookup(Call);
  if (!Description)
    return false;

  (Description->EvalFn)(this, Call, C);

  return C.isDifferent();
}

void StreamChecker::evalFopen(const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();
  const LocationContext *LCtx = C.getPredecessor()->getLocationContext();

  auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  DefinedSVal RetVal = SVB.conjureSymbolVal(nullptr, CE, LCtx, C.blockCount())
                           .castAs<DefinedSVal>();
  SymbolRef RetSym = RetVal.getAsSymbol();
  assert(RetSym && "RetVal must be a symbol here.");

  State = State->BindExpr(CE, C.getLocationContext(), RetVal);

  // Bifurcate the state into two: one with a valid FILE* pointer, the other
  // with a NULL.
  ProgramStateRef StateNotNull, StateNull;
  std::tie(StateNotNull, StateNull) =
      C.getConstraintManager().assumeDual(State, RetVal);

  StateNotNull = StateNotNull->set<StreamMap>(RetSym, StreamState::getOpened());
  StateNull = StateNull->set<StreamMap>(RetSym, StreamState::getOpenFailed());

  C.addTransition(StateNotNull);
  C.addTransition(StateNull);
}

void StreamChecker::evalFreopen(const CallEvent &Call,
                                CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  Optional<DefinedSVal> StreamVal = Call.getArgSVal(2).getAs<DefinedSVal>();
  if (!StreamVal)
    return;
  // Do not allow NULL as passed stream pointer.
  // This is not specified in the man page but may crash on some system.
  State = checkNullStream(*StreamVal, C, State);
  if (!State)
    return;

  SymbolRef StreamSym = StreamVal->getAsSymbol();
  // Do not care about special values for stream ("(FILE *)0x12345"?).
  if (!StreamSym)
    return;

  // Generate state for non-failed case.
  // Return value is the passed stream pointer.
  // According to the documentations, the stream is closed first
  // but any close error is ignored. The state changes to (or remains) opened.
  ProgramStateRef StateRetNotNull =
      State->BindExpr(CE, C.getLocationContext(), *StreamVal);
  // Generate state for NULL return value.
  // Stream switches to OpenFailed state.
  ProgramStateRef StateRetNull = State->BindExpr(CE, C.getLocationContext(),
                                                 C.getSValBuilder().makeNull());

  StateRetNotNull =
      StateRetNotNull->set<StreamMap>(StreamSym, StreamState::getOpened());
  StateRetNull =
      StateRetNull->set<StreamMap>(StreamSym, StreamState::getOpenFailed());

  C.addTransition(StateRetNotNull);
  C.addTransition(StateRetNull);
}

void StreamChecker::evalFclose(const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  State = checkDoubleClose(Call, C, State);
  if (State)
    C.addTransition(State);
}

void StreamChecker::evalFseek(const CallEvent &Call, CheckerContext &C) const {
  const Expr *AE2 = Call.getArgExpr(2);
  if (!AE2)
    return;

  ProgramStateRef State = C.getState();

  State = checkNullStream(Call.getArgSVal(0), C, State);
  if (!State)
    return;

  State =
      checkFseekWhence(State->getSVal(AE2, C.getLocationContext()), C, State);
  if (!State)
    return;

  C.addTransition(State);
}

void StreamChecker::checkArgNullStream(const CallEvent &Call, CheckerContext &C,
                                       unsigned ArgI) const {
  ProgramStateRef State = C.getState();
  State = checkNullStream(Call.getArgSVal(ArgI), C, State);
  if (State)
    C.addTransition(State);
}

ProgramStateRef StreamChecker::checkNullStream(SVal SV, CheckerContext &C,
                                               ProgramStateRef State) const {
  Optional<DefinedSVal> DV = SV.getAs<DefinedSVal>();
  if (!DV)
    return State;

  ConstraintManager &CM = C.getConstraintManager();
  ProgramStateRef StateNotNull, StateNull;
  std::tie(StateNotNull, StateNull) = CM.assumeDual(C.getState(), *DV);

  if (!StateNotNull && StateNull) {
    if (ExplodedNode *N = C.generateErrorNode(StateNull)) {
      if (!BT_nullfp)
        BT_nullfp.reset(new BuiltinBug(this, "NULL stream pointer",
                                       "Stream pointer might be NULL."));
      C.emitReport(std::make_unique<PathSensitiveBugReport>(
          *BT_nullfp, BT_nullfp->getDescription(), N));
    }
    return nullptr;
  }

  return StateNotNull;
}

// Check the legality of the 'whence' argument of 'fseek'.
ProgramStateRef StreamChecker::checkFseekWhence(SVal SV, CheckerContext &C,
                                                ProgramStateRef State) const {
  Optional<nonloc::ConcreteInt> CI = SV.getAs<nonloc::ConcreteInt>();
  if (!CI)
    return State;

  int64_t X = CI->getValue().getSExtValue();
  if (X >= 0 && X <= 2)
    return State;

  if (ExplodedNode *N = C.generateNonFatalErrorNode(State)) {
    if (!BT_illegalwhence)
      BT_illegalwhence.reset(
          new BuiltinBug(this, "Illegal whence argument",
                         "The whence argument to fseek() should be "
                         "SEEK_SET, SEEK_END, or SEEK_CUR."));
    C.emitReport(std::make_unique<PathSensitiveBugReport>(
        *BT_illegalwhence, BT_illegalwhence->getDescription(), N));
    return nullptr;
  }

  return State;
}

ProgramStateRef StreamChecker::checkDoubleClose(const CallEvent &Call,
                                                CheckerContext &C,
                                                ProgramStateRef State) const {
  SymbolRef Sym = Call.getArgSVal(0).getAsSymbol();
  if (!Sym)
    return State;

  const StreamState *SS = State->get<StreamMap>(Sym);

  // If the file stream is not tracked, return.
  if (!SS)
    return State;

  // Check: Double close a File Descriptor could cause undefined behaviour.
  // Conforming to man-pages
  if (SS->isClosed()) {
    ExplodedNode *N = C.generateErrorNode();
    if (N) {
      if (!BT_doubleclose)
        BT_doubleclose.reset(new BuiltinBug(
            this, "Double fclose", "Try to close a file Descriptor already"
                                   " closed. Cause undefined behaviour."));
      C.emitReport(std::make_unique<PathSensitiveBugReport>(
          *BT_doubleclose, BT_doubleclose->getDescription(), N));
      return nullptr;
    }

    return State;
  }

  // Close the File Descriptor.
  State = State->set<StreamMap>(Sym, StreamState::getClosed());

  return State;
}

void StreamChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  // TODO: Clean up the state.
  const StreamMapTy &Map = State->get<StreamMap>();
  for (const auto &I: Map) {
    SymbolRef Sym = I.first;
    const StreamState &SS = I.second;
    if (!SymReaper.isDead(Sym) || !SS.isOpened())
      continue;

    ExplodedNode *N = C.generateErrorNode();
    if (!N)
      continue;

    if (!BT_ResourceLeak)
      BT_ResourceLeak.reset(
          new BuiltinBug(this, "Resource Leak",
                         "Opened File never closed. Potential Resource leak."));
    C.emitReport(std::make_unique<PathSensitiveBugReport>(
        *BT_ResourceLeak, BT_ResourceLeak->getDescription(), N));
  }
}

void ento::registerStreamChecker(CheckerManager &mgr) {
  mgr.registerChecker<StreamChecker>();
}

bool ento::shouldRegisterStreamChecker(const LangOptions &LO) {
  return true;
}
