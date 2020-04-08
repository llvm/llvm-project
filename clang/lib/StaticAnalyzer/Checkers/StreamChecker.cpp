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

using namespace clang;
using namespace ento;

namespace {

/// Full state information about a stream pointer.
struct StreamState {
  /// State of a stream symbol.
  /// FIXME: We need maybe an "escaped" state later.
  enum KindTy {
    Opened, /// Stream is opened.
    Closed, /// Closed stream (an invalid stream pointer after it was closed).
    OpenFailed /// The last open operation has failed.
  } State;

  /// The error state of a stream.
  /// Valid only if the stream is opened.
  /// It is assumed that feof and ferror flags are never true at the same time.
  enum ErrorKindTy {
    /// No error flag is set (or stream is not open).
    NoError,
    /// EOF condition (`feof` is true).
    FEof,
    /// Other generic (non-EOF) error (`ferror` is true).
    FError,
  } ErrorState = NoError;

  bool isOpened() const { return State == Opened; }
  bool isClosed() const { return State == Closed; }
  bool isOpenFailed() const { return State == OpenFailed; }

  bool isNoError() const {
    assert(State == Opened && "Error undefined for closed stream.");
    return ErrorState == NoError;
  }
  bool isFEof() const {
    assert(State == Opened && "Error undefined for closed stream.");
    return ErrorState == FEof;
  }
  bool isFError() const {
    assert(State == Opened && "Error undefined for closed stream.");
    return ErrorState == FError;
  }

  bool operator==(const StreamState &X) const {
    // In not opened state error should always NoError.
    return State == X.State && ErrorState == X.ErrorState;
  }

  static StreamState getOpened() { return StreamState{Opened}; }
  static StreamState getClosed() { return StreamState{Closed}; }
  static StreamState getOpenFailed() { return StreamState{OpenFailed}; }
  static StreamState getOpenedWithFEof() { return StreamState{Opened, FEof}; }
  static StreamState getOpenedWithFError() {
    return StreamState{Opened, FError};
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(State);
    ID.AddInteger(ErrorState);
  }
};

class StreamChecker;
struct FnDescription;
using FnCheck = std::function<void(const StreamChecker *, const FnDescription *,
                                   const CallEvent &, CheckerContext &)>;

using ArgNoTy = unsigned int;
static const ArgNoTy ArgNone = std::numeric_limits<ArgNoTy>::max();

struct FnDescription {
  FnCheck PreFn;
  FnCheck EvalFn;
  ArgNoTy StreamArgNo;
};

/// Get the value of the stream argument out of the passed call event.
/// The call should contain a function that is described by Desc.
SVal getStreamArg(const FnDescription *Desc, const CallEvent &Call) {
  assert(Desc && Desc->StreamArgNo != ArgNone &&
         "Try to get a non-existing stream argument.");
  return Call.getArgSVal(Desc->StreamArgNo);
}

/// Create a conjured symbol return value for a call expression.
DefinedSVal makeRetVal(CheckerContext &C, const CallExpr *CE) {
  assert(CE && "Expecting a call expression.");

  const LocationContext *LCtx = C.getLocationContext();
  return C.getSValBuilder()
      .conjureSymbolVal(nullptr, CE, LCtx, C.blockCount())
      .castAs<DefinedSVal>();
}

class StreamChecker
    : public Checker<check::PreCall, eval::Call, check::DeadSymbols> {
  mutable std::unique_ptr<BuiltinBug> BT_nullfp, BT_illegalwhence,
      BT_UseAfterClose, BT_UseAfterOpenFailed, BT_ResourceLeak;

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;

  /// If true, evaluate special testing stream functions.
  bool TestMode = false;

private:
  CallDescriptionMap<FnDescription> FnDescriptions = {
      {{"fopen"}, {nullptr, &StreamChecker::evalFopen, ArgNone}},
      {{"freopen", 3},
       {&StreamChecker::preFreopen, &StreamChecker::evalFreopen, 2}},
      {{"tmpfile"}, {nullptr, &StreamChecker::evalFopen, ArgNone}},
      {{"fclose", 1},
       {&StreamChecker::preDefault, &StreamChecker::evalFclose, 0}},
      {{"fread", 4}, {&StreamChecker::preDefault, nullptr, 3}},
      {{"fwrite", 4}, {&StreamChecker::preDefault, nullptr, 3}},
      {{"fseek", 3}, {&StreamChecker::preFseek, nullptr, 0}},
      {{"ftell", 1}, {&StreamChecker::preDefault, nullptr, 0}},
      {{"rewind", 1}, {&StreamChecker::preDefault, nullptr, 0}},
      {{"fgetpos", 2}, {&StreamChecker::preDefault, nullptr, 0}},
      {{"fsetpos", 2}, {&StreamChecker::preDefault, nullptr, 0}},
      {{"clearerr", 1},
       {&StreamChecker::preDefault, &StreamChecker::evalClearerr, 0}},
      {{"feof", 1},
       {&StreamChecker::preDefault,
        &StreamChecker::evalFeofFerror<&StreamState::isFEof>, 0}},
      {{"ferror", 1},
       {&StreamChecker::preDefault,
        &StreamChecker::evalFeofFerror<&StreamState::isFError>, 0}},
      {{"fileno", 1}, {&StreamChecker::preDefault, nullptr, 0}},
  };

  CallDescriptionMap<FnDescription> FnTestDescriptions = {
      {{"StreamTesterChecker_make_feof_stream", 1},
       {nullptr,
        &StreamChecker::evalSetFeofFerror<&StreamState::getOpenedWithFEof>, 0}},
      {{"StreamTesterChecker_make_ferror_stream", 1},
       {nullptr,
        &StreamChecker::evalSetFeofFerror<&StreamState::getOpenedWithFError>,
        0}},
  };

  void evalFopen(const FnDescription *Desc, const CallEvent &Call,
                 CheckerContext &C) const;

  void preFreopen(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;
  void evalFreopen(const FnDescription *Desc, const CallEvent &Call,
                   CheckerContext &C) const;

  void evalFclose(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;

  void preFseek(const FnDescription *Desc, const CallEvent &Call,
                CheckerContext &C) const;

  void preDefault(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;

  void evalClearerr(const FnDescription *Desc, const CallEvent &Call,
                    CheckerContext &C) const;

  template <bool (StreamState::*IsOfError)() const>
  void evalFeofFerror(const FnDescription *Desc, const CallEvent &Call,
                      CheckerContext &C) const;

  template <StreamState (*GetState)()>
  void evalSetFeofFerror(const FnDescription *Desc, const CallEvent &Call,
                         CheckerContext &C) const;

  /// Check that the stream (in StreamVal) is not NULL.
  /// If it can only be NULL a fatal error is emitted and nullptr returned.
  /// Otherwise the return value is a new state where the stream is constrained
  /// to be non-null.
  ProgramStateRef ensureStreamNonNull(SVal StreamVal, CheckerContext &C,
                                      ProgramStateRef State) const;  

  /// Check that the stream is the opened state.
  /// If the stream is known to be not opened an error is generated
  /// and nullptr returned, otherwise the original state is returned.
  ProgramStateRef ensureStreamOpened(SVal StreamVal, CheckerContext &C,
                                     ProgramStateRef State) const;

  /// Check the legality of the 'whence' argument of 'fseek'.
  /// Generate error and return nullptr if it is found to be illegal.
  /// Otherwise returns the state.
  /// (State is not changed here because the "whence" value is already known.)
  ProgramStateRef ensureFseekWhenceCorrect(SVal WhenceVal, CheckerContext &C,
                                           ProgramStateRef State) const;  

  /// Find the description data of the function called by a call event.
  /// Returns nullptr if no function is recognized.
  const FnDescription *lookupFn(const CallEvent &Call) const {
    // Recognize "global C functions" with only integral or pointer arguments
    // (and matching name) as stream functions.
    if (!Call.isGlobalCFunction())
      return nullptr;
    for (auto P : Call.parameters()) {
      QualType T = P->getType();
      if (!T->isIntegralOrEnumerationType() && !T->isPointerType())
        return nullptr;
    }

    return FnDescriptions.lookup(Call);
  }  
};

} // end anonymous namespace

REGISTER_MAP_WITH_PROGRAMSTATE(StreamMap, SymbolRef, StreamState)

inline void assertStreamStateOpened(const StreamState *SS) {
  assert(SS->isOpened() &&
         "Previous create of error node for non-opened stream failed?");
}

void StreamChecker::checkPreCall(const CallEvent &Call,
                                 CheckerContext &C) const {
  const FnDescription *Desc = lookupFn(Call);
  if (!Desc || !Desc->PreFn)
    return;

  Desc->PreFn(this, Desc, Call, C);
}

bool StreamChecker::evalCall(const CallEvent &Call, CheckerContext &C) const {
  const FnDescription *Desc = lookupFn(Call);
  if (!Desc && TestMode)
    Desc = FnTestDescriptions.lookup(Call);
  if (!Desc || !Desc->EvalFn)
    return false;

  Desc->EvalFn(this, Desc, Call, C);

  return C.isDifferent();
}

void StreamChecker::evalFopen(const FnDescription *Desc, const CallEvent &Call,
                              CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const CallExpr *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  DefinedSVal RetVal = makeRetVal(C, CE);
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

void StreamChecker::preFreopen(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  // Do not allow NULL as passed stream pointer but allow a closed stream.
  ProgramStateRef State = C.getState();
  State = ensureStreamNonNull(getStreamArg(Desc, Call), C, State);
  if (!State)
    return;

  C.addTransition(State);
}

void StreamChecker::evalFreopen(const FnDescription *Desc,
                                const CallEvent &Call,
                                CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  Optional<DefinedSVal> StreamVal =
      getStreamArg(Desc, Call).getAs<DefinedSVal>();
  if (!StreamVal)
    return;

  SymbolRef StreamSym = StreamVal->getAsSymbol();
  // Do not care about concrete values for stream ("(FILE *)0x12345"?).
  // FIXME: Are stdin, stdout, stderr such values?
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

void StreamChecker::evalFclose(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef Sym = getStreamArg(Desc, Call).getAsSymbol();
  if (!Sym)
    return;

  const StreamState *SS = State->get<StreamMap>(Sym);
  if (!SS)
    return;

  assertStreamStateOpened(SS);

  // Close the File Descriptor.
  // Regardless if the close fails or not, stream becomes "closed"
  // and can not be used any more.
  State = State->set<StreamMap>(Sym, StreamState::getClosed());

  C.addTransition(State);
}

void StreamChecker::preFseek(const FnDescription *Desc, const CallEvent &Call,
                             CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SVal StreamVal = getStreamArg(Desc, Call);
  State = ensureStreamNonNull(StreamVal, C, State);
  if (!State)
    return;
  State = ensureStreamOpened(StreamVal, C, State);
  if (!State)
    return;
  State = ensureFseekWhenceCorrect(Call.getArgSVal(2), C, State);
  if (!State)
    return;

  C.addTransition(State);
}

void StreamChecker::evalClearerr(const FnDescription *Desc,
                                 const CallEvent &Call,
                                 CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef StreamSym = getStreamArg(Desc, Call).getAsSymbol();
  if (!StreamSym)
    return;

  const StreamState *SS = State->get<StreamMap>(StreamSym);
  if (!SS)
    return;

  assertStreamStateOpened(SS);

  if (SS->isNoError())
    return;

  State = State->set<StreamMap>(StreamSym, StreamState::getOpened());
  C.addTransition(State);
}

template <bool (StreamState::*IsOfError)() const>
void StreamChecker::evalFeofFerror(const FnDescription *Desc,
                                   const CallEvent &Call,
                                   CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef StreamSym = getStreamArg(Desc, Call).getAsSymbol();
  if (!StreamSym)
    return;

  const CallExpr *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  const StreamState *SS = State->get<StreamMap>(StreamSym);
  // Ignore the call if the stream is not tracked.
  if (!SS)
    return;

  assertStreamStateOpened(SS);

  if ((SS->*IsOfError)()) {
    // Function returns nonzero.
    DefinedSVal RetVal = makeRetVal(C, CE);
    State = State->BindExpr(CE, C.getLocationContext(), RetVal);
    State = State->assume(RetVal, true);
    assert(State && "Assumption on new value should not fail.");
  } else {
    // Return zero.
    State = State->BindExpr(CE, C.getLocationContext(),
                            C.getSValBuilder().makeIntVal(0, false));
  }
  C.addTransition(State);
}

void StreamChecker::preDefault(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SVal StreamVal = getStreamArg(Desc, Call);
  State = ensureStreamNonNull(StreamVal, C, State);
  if (!State)
    return;
  State = ensureStreamOpened(StreamVal, C, State);
  if (!State)
    return;

  C.addTransition(State);
}

template <StreamState (*GetState)()>
void StreamChecker::evalSetFeofFerror(const FnDescription *Desc,
                                      const CallEvent &Call,
                                      CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef StreamSym = getStreamArg(Desc, Call).getAsSymbol();
  assert(StreamSym && "Operation not permitted on non-symbolic stream value.");
  State = State->set<StreamMap>(StreamSym, (*GetState)());
  C.addTransition(State);
}

ProgramStateRef
StreamChecker::ensureStreamNonNull(SVal StreamVal, CheckerContext &C,
                                   ProgramStateRef State) const {
  auto Stream = StreamVal.getAs<DefinedSVal>();
  if (!Stream)
    return State;

  ConstraintManager &CM = C.getConstraintManager();

  ProgramStateRef StateNotNull, StateNull;
  std::tie(StateNotNull, StateNull) = CM.assumeDual(C.getState(), *Stream);

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

ProgramStateRef StreamChecker::ensureStreamOpened(SVal StreamVal,
                                                  CheckerContext &C,
                                                  ProgramStateRef State) const {
  SymbolRef Sym = StreamVal.getAsSymbol();
  if (!Sym)
    return State;

  const StreamState *SS = State->get<StreamMap>(Sym);
  if (!SS)
    return State;

  if (SS->isClosed()) {
    // Using a stream pointer after 'fclose' causes undefined behavior
    // according to cppreference.com .
    ExplodedNode *N = C.generateErrorNode();
    if (N) {
      if (!BT_UseAfterClose)
        BT_UseAfterClose.reset(new BuiltinBug(this, "Closed stream",
                                              "Stream might be already closed. "
                                              "Causes undefined behaviour."));
      C.emitReport(std::make_unique<PathSensitiveBugReport>(
          *BT_UseAfterClose, BT_UseAfterClose->getDescription(), N));
      return nullptr;
    }

    return State;
  }

  if (SS->isOpenFailed()) {
    // Using a stream that has failed to open is likely to cause problems.
    // This should usually not occur because stream pointer is NULL.
    // But freopen can cause a state when stream pointer remains non-null but
    // failed to open.
    ExplodedNode *N = C.generateErrorNode();
    if (N) {
      if (!BT_UseAfterOpenFailed)
        BT_UseAfterOpenFailed.reset(
            new BuiltinBug(this, "Invalid stream",
                           "Stream might be invalid after "
                           "(re-)opening it has failed. "
                           "Can cause undefined behaviour."));
      C.emitReport(std::make_unique<PathSensitiveBugReport>(
          *BT_UseAfterOpenFailed, BT_UseAfterOpenFailed->getDescription(), N));
      return nullptr;
    }
    return State;
  }

  return State;
}

ProgramStateRef
StreamChecker::ensureFseekWhenceCorrect(SVal WhenceVal, CheckerContext &C,
                                        ProgramStateRef State) const {
  Optional<nonloc::ConcreteInt> CI = WhenceVal.getAs<nonloc::ConcreteInt>();
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

void StreamChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  // TODO: Clean up the state.
  const StreamMapTy &Map = State->get<StreamMap>();
  for (const auto &I : Map) {
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

void ento::registerStreamChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<StreamChecker>();
}

bool ento::shouldRegisterStreamChecker(const CheckerManager &Mgr) {
  return true;
}

void ento::registerStreamTesterChecker(CheckerManager &Mgr) {
  auto *Checker = Mgr.getChecker<StreamChecker>();
  Checker->TestMode = true;
}

bool ento::shouldRegisterStreamTesterChecker(const CheckerManager &Mgr) {
  return true;
}

