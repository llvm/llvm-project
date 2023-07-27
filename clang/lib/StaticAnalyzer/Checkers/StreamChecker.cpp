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
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include <functional>
#include <optional>

using namespace clang;
using namespace ento;
using namespace std::placeholders;

//===----------------------------------------------------------------------===//
// Definition of state data structures.
//===----------------------------------------------------------------------===//

namespace {

struct FnDescription;

/// State of the stream error flags.
/// Sometimes it is not known to the checker what error flags are set.
/// This is indicated by setting more than one flag to true.
/// This is an optimization to avoid state splits.
/// A stream can either be in FEOF or FERROR but not both at the same time.
/// Multiple flags are set to handle the corresponding states together.
struct StreamErrorState {
  /// The stream can be in state where none of the error flags set.
  bool NoError = true;
  /// The stream can be in state where the EOF indicator is set.
  bool FEof = false;
  /// The stream can be in state where the error indicator is set.
  bool FError = false;

  bool isNoError() const { return NoError && !FEof && !FError; }
  bool isFEof() const { return !NoError && FEof && !FError; }
  bool isFError() const { return !NoError && !FEof && FError; }

  bool operator==(const StreamErrorState &ES) const {
    return NoError == ES.NoError && FEof == ES.FEof && FError == ES.FError;
  }

  bool operator!=(const StreamErrorState &ES) const { return !(*this == ES); }

  StreamErrorState operator|(const StreamErrorState &E) const {
    return {NoError || E.NoError, FEof || E.FEof, FError || E.FError};
  }

  StreamErrorState operator&(const StreamErrorState &E) const {
    return {NoError && E.NoError, FEof && E.FEof, FError && E.FError};
  }

  StreamErrorState operator~() const { return {!NoError, !FEof, !FError}; }

  /// Returns if the StreamErrorState is a valid object.
  operator bool() const { return NoError || FEof || FError; }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddBoolean(NoError);
    ID.AddBoolean(FEof);
    ID.AddBoolean(FError);
  }
};

const StreamErrorState ErrorNone{true, false, false};
const StreamErrorState ErrorFEof{false, true, false};
const StreamErrorState ErrorFError{false, false, true};

/// Full state information about a stream pointer.
struct StreamState {
  /// The last file operation called in the stream.
  /// Can be nullptr.
  const FnDescription *LastOperation;

  /// State of a stream symbol.
  enum KindTy {
    Opened, /// Stream is opened.
    Closed, /// Closed stream (an invalid stream pointer after it was closed).
    OpenFailed /// The last open operation has failed.
  } State;

  /// State of the error flags.
  /// Ignored in non-opened stream state but must be NoError.
  StreamErrorState const ErrorState;

  /// Indicate if the file has an "indeterminate file position indicator".
  /// This can be set at a failing read or write or seek operation.
  /// If it is set no more read or write is allowed.
  /// This value is not dependent on the stream error flags:
  /// The error flag may be cleared with `clearerr` but the file position
  /// remains still indeterminate.
  /// This value applies to all error states in ErrorState except FEOF.
  /// An EOF+indeterminate state is the same as EOF state.
  bool const FilePositionIndeterminate = false;

  StreamState(const FnDescription *L, KindTy S, const StreamErrorState &ES,
              bool IsFilePositionIndeterminate)
      : LastOperation(L), State(S), ErrorState(ES),
        FilePositionIndeterminate(IsFilePositionIndeterminate) {
    assert((!ES.isFEof() || !IsFilePositionIndeterminate) &&
           "FilePositionIndeterminate should be false in FEof case.");
    assert((State == Opened || ErrorState.isNoError()) &&
           "ErrorState should be None in non-opened stream state.");
  }

  bool isOpened() const { return State == Opened; }
  bool isClosed() const { return State == Closed; }
  bool isOpenFailed() const { return State == OpenFailed; }

  bool operator==(const StreamState &X) const {
    // In not opened state error state should always NoError, so comparison
    // here is no problem.
    return LastOperation == X.LastOperation && State == X.State &&
           ErrorState == X.ErrorState &&
           FilePositionIndeterminate == X.FilePositionIndeterminate;
  }

  static StreamState getOpened(const FnDescription *L,
                               const StreamErrorState &ES = ErrorNone,
                               bool IsFilePositionIndeterminate = false) {
    return StreamState{L, Opened, ES, IsFilePositionIndeterminate};
  }
  static StreamState getClosed(const FnDescription *L) {
    return StreamState{L, Closed, {}, false};
  }
  static StreamState getOpenFailed(const FnDescription *L) {
    return StreamState{L, OpenFailed, {}, false};
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(LastOperation);
    ID.AddInteger(State);
    ErrorState.Profile(ID);
    ID.AddBoolean(FilePositionIndeterminate);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// StreamChecker class and utility functions.
//===----------------------------------------------------------------------===//

namespace {

class StreamChecker;
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

ProgramStateRef bindAndAssumeTrue(ProgramStateRef State, CheckerContext &C,
                                  const CallExpr *CE) {
  DefinedSVal RetVal = makeRetVal(C, CE);
  State = State->BindExpr(CE, C.getLocationContext(), RetVal);
  State = State->assume(RetVal, true);
  assert(State && "Assumption on new value should not fail.");
  return State;
}

ProgramStateRef bindInt(uint64_t Value, ProgramStateRef State,
                        CheckerContext &C, const CallExpr *CE) {
  State = State->BindExpr(CE, C.getLocationContext(),
                          C.getSValBuilder().makeIntVal(Value, CE->getType()));
  return State;
}

class StreamChecker : public Checker<check::PreCall, eval::Call,
                                     check::DeadSymbols, check::PointerEscape> {
  BugType BT_FileNull{this, "NULL stream pointer", "Stream handling error"};
  BugType BT_UseAfterClose{this, "Closed stream", "Stream handling error"};
  BugType BT_UseAfterOpenFailed{this, "Invalid stream",
                                "Stream handling error"};
  BugType BT_IndeterminatePosition{this, "Invalid stream state",
                                   "Stream handling error"};
  BugType BT_IllegalWhence{this, "Illegal whence argument",
                           "Stream handling error"};
  BugType BT_StreamEof{this, "Stream already in EOF", "Stream handling error"};
  BugType BT_ResourceLeak{this, "Resource leak", "Stream handling error",
                          /*SuppressOnSink =*/true};

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  ProgramStateRef checkPointerEscape(ProgramStateRef State,
                                     const InvalidatedSymbols &Escaped,
                                     const CallEvent *Call,
                                     PointerEscapeKind Kind) const;

  /// If true, evaluate special testing stream functions.
  bool TestMode = false;

  const BugType *getBT_StreamEof() const { return &BT_StreamEof; }

private:
  CallDescriptionMap<FnDescription> FnDescriptions = {
      {{{"fopen"}}, {nullptr, &StreamChecker::evalFopen, ArgNone}},
      {{{"freopen"}, 3},
       {&StreamChecker::preFreopen, &StreamChecker::evalFreopen, 2}},
      {{{"tmpfile"}}, {nullptr, &StreamChecker::evalFopen, ArgNone}},
      {{{"fclose"}, 1},
       {&StreamChecker::preDefault, &StreamChecker::evalFclose, 0}},
      {{{"fread"}, 4},
       {&StreamChecker::preFread,
        std::bind(&StreamChecker::evalFreadFwrite, _1, _2, _3, _4, true), 3}},
      {{{"fwrite"}, 4},
       {&StreamChecker::preFwrite,
        std::bind(&StreamChecker::evalFreadFwrite, _1, _2, _3, _4, false), 3}},
      {{{"fseek"}, 3},
       {&StreamChecker::preFseek, &StreamChecker::evalFseek, 0}},
      {{{"ftell"}, 1},
       {&StreamChecker::preDefault, &StreamChecker::evalFtell, 0}},
      {{{"rewind"}, 1},
       {&StreamChecker::preDefault, &StreamChecker::evalRewind, 0}},
      {{{"fgetpos"}, 2},
       {&StreamChecker::preDefault, &StreamChecker::evalFgetpos, 0}},
      {{{"fsetpos"}, 2},
       {&StreamChecker::preDefault, &StreamChecker::evalFsetpos, 0}},
      {{{"clearerr"}, 1},
       {&StreamChecker::preDefault, &StreamChecker::evalClearerr, 0}},
      {{{"feof"}, 1},
       {&StreamChecker::preDefault,
        std::bind(&StreamChecker::evalFeofFerror, _1, _2, _3, _4, ErrorFEof),
        0}},
      {{{"ferror"}, 1},
       {&StreamChecker::preDefault,
        std::bind(&StreamChecker::evalFeofFerror, _1, _2, _3, _4, ErrorFError),
        0}},
      {{{"fileno"}, 1}, {&StreamChecker::preDefault, nullptr, 0}},
  };

  CallDescriptionMap<FnDescription> FnTestDescriptions = {
      {{{"StreamTesterChecker_make_feof_stream"}, 1},
       {nullptr,
        std::bind(&StreamChecker::evalSetFeofFerror, _1, _2, _3, _4, ErrorFEof),
        0}},
      {{{"StreamTesterChecker_make_ferror_stream"}, 1},
       {nullptr,
        std::bind(&StreamChecker::evalSetFeofFerror, _1, _2, _3, _4,
                  ErrorFError),
        0}},
  };

  /// Expanded value of EOF, empty before initialization.
  mutable std::optional<int> EofVal;
  /// Expanded value of SEEK_SET, 0 if not found.
  mutable int SeekSetVal = 0;
  /// Expanded value of SEEK_CUR, 1 if not found.
  mutable int SeekCurVal = 1;
  /// Expanded value of SEEK_END, 2 if not found.
  mutable int SeekEndVal = 2;

  void evalFopen(const FnDescription *Desc, const CallEvent &Call,
                 CheckerContext &C) const;

  void preFreopen(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;
  void evalFreopen(const FnDescription *Desc, const CallEvent &Call,
                   CheckerContext &C) const;

  void evalFclose(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;

  void preFread(const FnDescription *Desc, const CallEvent &Call,
                CheckerContext &C) const;

  void preFwrite(const FnDescription *Desc, const CallEvent &Call,
                 CheckerContext &C) const;

  void evalFreadFwrite(const FnDescription *Desc, const CallEvent &Call,
                       CheckerContext &C, bool IsFread) const;

  void preFseek(const FnDescription *Desc, const CallEvent &Call,
                CheckerContext &C) const;
  void evalFseek(const FnDescription *Desc, const CallEvent &Call,
                 CheckerContext &C) const;

  void evalFgetpos(const FnDescription *Desc, const CallEvent &Call,
                   CheckerContext &C) const;

  void evalFsetpos(const FnDescription *Desc, const CallEvent &Call,
                   CheckerContext &C) const;

  void evalFtell(const FnDescription *Desc, const CallEvent &Call,
                 CheckerContext &C) const;

  void evalRewind(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;

  void preDefault(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;

  void evalClearerr(const FnDescription *Desc, const CallEvent &Call,
                    CheckerContext &C) const;

  void evalFeofFerror(const FnDescription *Desc, const CallEvent &Call,
                      CheckerContext &C,
                      const StreamErrorState &ErrorKind) const;

  void evalSetFeofFerror(const FnDescription *Desc, const CallEvent &Call,
                         CheckerContext &C,
                         const StreamErrorState &ErrorKind) const;

  /// Check that the stream (in StreamVal) is not NULL.
  /// If it can only be NULL a fatal error is emitted and nullptr returned.
  /// Otherwise the return value is a new state where the stream is constrained
  /// to be non-null.
  ProgramStateRef ensureStreamNonNull(SVal StreamVal, const Expr *StreamE,
                                      CheckerContext &C,
                                      ProgramStateRef State) const;

  /// Check that the stream is the opened state.
  /// If the stream is known to be not opened an error is generated
  /// and nullptr returned, otherwise the original state is returned.
  ProgramStateRef ensureStreamOpened(SVal StreamVal, CheckerContext &C,
                                     ProgramStateRef State) const;

  /// Check that the stream has not an invalid ("indeterminate") file position,
  /// generate warning for it.
  /// (EOF is not an invalid position.)
  /// The returned state can be nullptr if a fatal error was generated.
  /// It can return non-null state if the stream has not an invalid position or
  /// there is execution path with non-invalid position.
  ProgramStateRef
  ensureNoFilePositionIndeterminate(SVal StreamVal, CheckerContext &C,
                                    ProgramStateRef State) const;

  /// Check the legality of the 'whence' argument of 'fseek'.
  /// Generate error and return nullptr if it is found to be illegal.
  /// Otherwise returns the state.
  /// (State is not changed here because the "whence" value is already known.)
  ProgramStateRef ensureFseekWhenceCorrect(SVal WhenceVal, CheckerContext &C,
                                           ProgramStateRef State) const;

  /// Generate warning about stream in EOF state.
  /// There will be always a state transition into the passed State,
  /// by the new non-fatal error node or (if failed) a normal transition,
  /// to ensure uniform handling.
  void reportFEofWarning(SymbolRef StreamSym, CheckerContext &C,
                         ProgramStateRef State) const;

  /// Emit resource leak warnings for the given symbols.
  /// Createn a non-fatal error node for these, and returns it (if any warnings
  /// were generated). Return value is non-null.
  ExplodedNode *reportLeaks(const SmallVector<SymbolRef, 2> &LeakedSyms,
                            CheckerContext &C, ExplodedNode *Pred) const;

  /// Find the description data of the function called by a call event.
  /// Returns nullptr if no function is recognized.
  const FnDescription *lookupFn(const CallEvent &Call) const {
    // Recognize "global C functions" with only integral or pointer arguments
    // (and matching name) as stream functions.
    if (!Call.isGlobalCFunction())
      return nullptr;
    for (auto *P : Call.parameters()) {
      QualType T = P->getType();
      if (!T->isIntegralOrEnumerationType() && !T->isPointerType())
        return nullptr;
    }

    return FnDescriptions.lookup(Call);
  }

  /// Generate a message for BugReporterVisitor if the stored symbol is
  /// marked as interesting by the actual bug report.
  // FIXME: Use lambda instead.
  struct NoteFn {
    const BugType *BT_ResourceLeak;
    SymbolRef StreamSym;
    std::string Message;

    std::string operator()(PathSensitiveBugReport &BR) const {
      if (BR.isInteresting(StreamSym) && &BR.getBugType() == BT_ResourceLeak)
        return Message;

      return "";
    }
  };

  const NoteTag *constructNoteTag(CheckerContext &C, SymbolRef StreamSym,
                                  const std::string &Message) const {
    return C.getNoteTag(NoteFn{&BT_ResourceLeak, StreamSym, Message});
  }

  const NoteTag *constructSetEofNoteTag(CheckerContext &C,
                                        SymbolRef StreamSym) const {
    return C.getNoteTag([this, StreamSym](PathSensitiveBugReport &BR) {
      if (!BR.isInteresting(StreamSym) ||
          &BR.getBugType() != this->getBT_StreamEof())
        return "";

      BR.markNotInteresting(StreamSym);

      return "Assuming stream reaches end-of-file here";
    });
  }

  void initMacroValues(CheckerContext &C) const {
    if (EofVal)
      return;

    if (const std::optional<int> OptInt =
            tryExpandAsInteger("EOF", C.getPreprocessor()))
      EofVal = *OptInt;
    else
      EofVal = -1;
    if (const std::optional<int> OptInt =
            tryExpandAsInteger("SEEK_SET", C.getPreprocessor()))
      SeekSetVal = *OptInt;
    if (const std::optional<int> OptInt =
            tryExpandAsInteger("SEEK_END", C.getPreprocessor()))
      SeekEndVal = *OptInt;
    if (const std::optional<int> OptInt =
            tryExpandAsInteger("SEEK_CUR", C.getPreprocessor()))
      SeekCurVal = *OptInt;
  }

  /// Searches for the ExplodedNode where the file descriptor was acquired for
  /// StreamSym.
  static const ExplodedNode *getAcquisitionSite(const ExplodedNode *N,
                                                SymbolRef StreamSym,
                                                CheckerContext &C);
};

} // end anonymous namespace

// This map holds the state of a stream.
// The stream is identified with a SymbolRef that is created when a stream
// opening function is modeled by the checker.
REGISTER_MAP_WITH_PROGRAMSTATE(StreamMap, SymbolRef, StreamState)

inline void assertStreamStateOpened(const StreamState *SS) {
  assert(SS->isOpened() && "Stream is expected to be opened");
}

const ExplodedNode *StreamChecker::getAcquisitionSite(const ExplodedNode *N,
                                                      SymbolRef StreamSym,
                                                      CheckerContext &C) {
  ProgramStateRef State = N->getState();
  // When bug type is resource leak, exploded node N may not have state info
  // for leaked file descriptor, but predecessor should have it.
  if (!State->get<StreamMap>(StreamSym))
    N = N->getFirstPred();

  const ExplodedNode *Pred = N;
  while (N) {
    State = N->getState();
    if (!State->get<StreamMap>(StreamSym))
      return Pred;
    Pred = N;
    N = N->getFirstPred();
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Methods of StreamChecker.
//===----------------------------------------------------------------------===//

void StreamChecker::checkPreCall(const CallEvent &Call,
                                 CheckerContext &C) const {
  initMacroValues(C);

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

  StateNotNull =
      StateNotNull->set<StreamMap>(RetSym, StreamState::getOpened(Desc));
  StateNull =
      StateNull->set<StreamMap>(RetSym, StreamState::getOpenFailed(Desc));

  C.addTransition(StateNotNull,
                  constructNoteTag(C, RetSym, "Stream opened here"));
  C.addTransition(StateNull);
}

void StreamChecker::preFreopen(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  // Do not allow NULL as passed stream pointer but allow a closed stream.
  ProgramStateRef State = C.getState();
  State = ensureStreamNonNull(getStreamArg(Desc, Call),
                              Call.getArgExpr(Desc->StreamArgNo), C, State);
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

  std::optional<DefinedSVal> StreamVal =
      getStreamArg(Desc, Call).getAs<DefinedSVal>();
  if (!StreamVal)
    return;

  SymbolRef StreamSym = StreamVal->getAsSymbol();
  // Do not care about concrete values for stream ("(FILE *)0x12345"?).
  // FIXME: Can be stdin, stdout, stderr such values?
  if (!StreamSym)
    return;

  // Do not handle untracked stream. It is probably escaped.
  if (!State->get<StreamMap>(StreamSym))
    return;

  // Generate state for non-failed case.
  // Return value is the passed stream pointer.
  // According to the documentations, the stream is closed first
  // but any close error is ignored. The state changes to (or remains) opened.
  ProgramStateRef StateRetNotNull =
      State->BindExpr(CE, C.getLocationContext(), *StreamVal);
  // Generate state for NULL return value.
  // Stream switches to OpenFailed state.
  ProgramStateRef StateRetNull =
      State->BindExpr(CE, C.getLocationContext(),
                      C.getSValBuilder().makeNullWithType(CE->getType()));

  StateRetNotNull =
      StateRetNotNull->set<StreamMap>(StreamSym, StreamState::getOpened(Desc));
  StateRetNull =
      StateRetNull->set<StreamMap>(StreamSym, StreamState::getOpenFailed(Desc));

  C.addTransition(StateRetNotNull,
                  constructNoteTag(C, StreamSym, "Stream reopened here"));
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

  auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  assertStreamStateOpened(SS);

  // Close the File Descriptor.
  // Regardless if the close fails or not, stream becomes "closed"
  // and can not be used any more.
  State = State->set<StreamMap>(Sym, StreamState::getClosed(Desc));

  // Return 0 on success, EOF on failure.
  SValBuilder &SVB = C.getSValBuilder();
  ProgramStateRef StateSuccess = State->BindExpr(
      CE, C.getLocationContext(), SVB.makeIntVal(0, C.getASTContext().IntTy));
  ProgramStateRef StateFailure =
      State->BindExpr(CE, C.getLocationContext(),
                      SVB.makeIntVal(*EofVal, C.getASTContext().IntTy));

  C.addTransition(StateSuccess);
  C.addTransition(StateFailure);
}

void StreamChecker::preFread(const FnDescription *Desc, const CallEvent &Call,
                             CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SVal StreamVal = getStreamArg(Desc, Call);
  State = ensureStreamNonNull(StreamVal, Call.getArgExpr(Desc->StreamArgNo), C,
                              State);
  if (!State)
    return;
  State = ensureStreamOpened(StreamVal, C, State);
  if (!State)
    return;
  State = ensureNoFilePositionIndeterminate(StreamVal, C, State);
  if (!State)
    return;

  SymbolRef Sym = StreamVal.getAsSymbol();
  if (Sym && State->get<StreamMap>(Sym)) {
    const StreamState *SS = State->get<StreamMap>(Sym);
    if (SS->ErrorState & ErrorFEof)
      reportFEofWarning(Sym, C, State);
  } else {
    C.addTransition(State);
  }
}

void StreamChecker::preFwrite(const FnDescription *Desc, const CallEvent &Call,
                              CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SVal StreamVal = getStreamArg(Desc, Call);
  State = ensureStreamNonNull(StreamVal, Call.getArgExpr(Desc->StreamArgNo), C,
                              State);
  if (!State)
    return;
  State = ensureStreamOpened(StreamVal, C, State);
  if (!State)
    return;
  State = ensureNoFilePositionIndeterminate(StreamVal, C, State);
  if (!State)
    return;

  C.addTransition(State);
}

void StreamChecker::evalFreadFwrite(const FnDescription *Desc,
                                    const CallEvent &Call, CheckerContext &C,
                                    bool IsFread) const {
  ProgramStateRef State = C.getState();
  SymbolRef StreamSym = getStreamArg(Desc, Call).getAsSymbol();
  if (!StreamSym)
    return;

  const CallExpr *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  std::optional<NonLoc> SizeVal = Call.getArgSVal(1).getAs<NonLoc>();
  if (!SizeVal)
    return;
  std::optional<NonLoc> NMembVal = Call.getArgSVal(2).getAs<NonLoc>();
  if (!NMembVal)
    return;

  const StreamState *OldSS = State->get<StreamMap>(StreamSym);
  if (!OldSS)
    return;

  assertStreamStateOpened(OldSS);

  // C'99 standard, §7.19.8.1.3, the return value of fread:
  // The fread function returns the number of elements successfully read, which
  // may be less than nmemb if a read error or end-of-file is encountered. If
  // size or nmemb is zero, fread returns zero and the contents of the array and
  // the state of the stream remain unchanged.

  if (State->isNull(*SizeVal).isConstrainedTrue() ||
      State->isNull(*NMembVal).isConstrainedTrue()) {
    // This is the "size or nmemb is zero" case.
    // Just return 0, do nothing more (not clear the error flags).
    State = bindInt(0, State, C, CE);
    C.addTransition(State);
    return;
  }

  // Generate a transition for the success state.
  // If we know the state to be FEOF at fread, do not add a success state.
  if (!IsFread || (OldSS->ErrorState != ErrorFEof)) {
    ProgramStateRef StateNotFailed =
        State->BindExpr(CE, C.getLocationContext(), *NMembVal);
    StateNotFailed =
        StateNotFailed->set<StreamMap>(StreamSym, StreamState::getOpened(Desc));
    C.addTransition(StateNotFailed);
  }

  // Add transition for the failed state.
  NonLoc RetVal = makeRetVal(C, CE).castAs<NonLoc>();
  ProgramStateRef StateFailed =
      State->BindExpr(CE, C.getLocationContext(), RetVal);
  auto Cond =
      C.getSValBuilder()
          .evalBinOpNN(State, BO_LT, RetVal, *NMembVal, C.getASTContext().IntTy)
          .getAs<DefinedOrUnknownSVal>();
  if (!Cond)
    return;
  StateFailed = StateFailed->assume(*Cond, true);
  if (!StateFailed)
    return;

  StreamErrorState NewES;
  if (IsFread)
    NewES =
        (OldSS->ErrorState == ErrorFEof) ? ErrorFEof : ErrorFEof | ErrorFError;
  else
    NewES = ErrorFError;
  // If a (non-EOF) error occurs, the resulting value of the file position
  // indicator for the stream is indeterminate.
  StreamState NewSS = StreamState::getOpened(Desc, NewES, !NewES.isFEof());
  StateFailed = StateFailed->set<StreamMap>(StreamSym, NewSS);
  if (IsFread && OldSS->ErrorState != ErrorFEof)
    C.addTransition(StateFailed, constructSetEofNoteTag(C, StreamSym));
  else
    C.addTransition(StateFailed);
}

void StreamChecker::preFseek(const FnDescription *Desc, const CallEvent &Call,
                             CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SVal StreamVal = getStreamArg(Desc, Call);
  State = ensureStreamNonNull(StreamVal, Call.getArgExpr(Desc->StreamArgNo), C,
                              State);
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

void StreamChecker::evalFseek(const FnDescription *Desc, const CallEvent &Call,
                              CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef StreamSym = getStreamArg(Desc, Call).getAsSymbol();
  if (!StreamSym)
    return;

  const CallExpr *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  // Ignore the call if the stream is not tracked.
  if (!State->get<StreamMap>(StreamSym))
    return;

  const llvm::APSInt *PosV =
      C.getSValBuilder().getKnownValue(State, Call.getArgSVal(1));
  const llvm::APSInt *WhenceV =
      C.getSValBuilder().getKnownValue(State, Call.getArgSVal(2));

  DefinedSVal RetVal = makeRetVal(C, CE);

  // Make expression result.
  State = State->BindExpr(CE, C.getLocationContext(), RetVal);

  // Bifurcate the state into failed and non-failed.
  // Return zero on success, nonzero on error.
  ProgramStateRef StateNotFailed, StateFailed;
  std::tie(StateFailed, StateNotFailed) =
      C.getConstraintManager().assumeDual(State, RetVal);

  // Reset the state to opened with no error.
  StateNotFailed =
      StateNotFailed->set<StreamMap>(StreamSym, StreamState::getOpened(Desc));
  // We get error.
  // It is possible that fseek fails but sets none of the error flags.
  // If fseek failed, assume that the file position becomes indeterminate in any
  // case.
  StreamErrorState NewErrS = ErrorNone | ErrorFError;
  // Setting the position to start of file never produces EOF error.
  if (!(PosV && *PosV == 0 && WhenceV && *WhenceV == SeekSetVal))
    NewErrS = NewErrS | ErrorFEof;
  StateFailed = StateFailed->set<StreamMap>(
      StreamSym, StreamState::getOpened(Desc, NewErrS, true));

  C.addTransition(StateNotFailed);
  C.addTransition(StateFailed, constructSetEofNoteTag(C, StreamSym));
}

void StreamChecker::evalFgetpos(const FnDescription *Desc,
                                const CallEvent &Call,
                                CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef Sym = getStreamArg(Desc, Call).getAsSymbol();
  if (!Sym)
    return;

  // Do not evaluate if stream is not found.
  if (!State->get<StreamMap>(Sym))
    return;

  auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  DefinedSVal RetVal = makeRetVal(C, CE);
  State = State->BindExpr(CE, C.getLocationContext(), RetVal);
  ProgramStateRef StateNotFailed, StateFailed;
  std::tie(StateFailed, StateNotFailed) =
      C.getConstraintManager().assumeDual(State, RetVal);

  // This function does not affect the stream state.
  // Still we add success and failure state with the appropriate return value.
  // StdLibraryFunctionsChecker can change these states (set the 'errno' state).
  C.addTransition(StateNotFailed);
  C.addTransition(StateFailed);
}

void StreamChecker::evalFsetpos(const FnDescription *Desc,
                                const CallEvent &Call,
                                CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef StreamSym = getStreamArg(Desc, Call).getAsSymbol();
  if (!StreamSym)
    return;

  const StreamState *SS = State->get<StreamMap>(StreamSym);
  if (!SS)
    return;

  auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  assertStreamStateOpened(SS);

  DefinedSVal RetVal = makeRetVal(C, CE);
  State = State->BindExpr(CE, C.getLocationContext(), RetVal);
  ProgramStateRef StateNotFailed, StateFailed;
  std::tie(StateFailed, StateNotFailed) =
      C.getConstraintManager().assumeDual(State, RetVal);

  StateNotFailed = StateNotFailed->set<StreamMap>(
      StreamSym, StreamState::getOpened(Desc, ErrorNone, false));

  // At failure ferror could be set.
  // The standards do not tell what happens with the file position at failure.
  // But we can assume that it is dangerous to make a next I/O operation after
  // the position was not set correctly (similar to 'fseek').
  StateFailed = StateFailed->set<StreamMap>(
      StreamSym, StreamState::getOpened(Desc, ErrorNone | ErrorFError, true));

  C.addTransition(StateNotFailed);
  C.addTransition(StateFailed);
}

void StreamChecker::evalFtell(const FnDescription *Desc, const CallEvent &Call,
                              CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef Sym = getStreamArg(Desc, Call).getAsSymbol();
  if (!Sym)
    return;

  if (!State->get<StreamMap>(Sym))
    return;

  auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  SValBuilder &SVB = C.getSValBuilder();
  NonLoc RetVal = makeRetVal(C, CE).castAs<NonLoc>();
  ProgramStateRef StateNotFailed =
      State->BindExpr(CE, C.getLocationContext(), RetVal);
  auto Cond = SVB.evalBinOp(State, BO_GE, RetVal,
                            SVB.makeZeroVal(C.getASTContext().LongTy),
                            SVB.getConditionType())
                  .getAs<DefinedOrUnknownSVal>();
  if (!Cond)
    return;
  StateNotFailed = StateNotFailed->assume(*Cond, true);
  if (!StateNotFailed)
    return;

  ProgramStateRef StateFailed = State->BindExpr(
      CE, C.getLocationContext(), SVB.makeIntVal(-1, C.getASTContext().LongTy));

  C.addTransition(StateNotFailed);
  C.addTransition(StateFailed);
}

void StreamChecker::evalRewind(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef StreamSym = getStreamArg(Desc, Call).getAsSymbol();
  if (!StreamSym)
    return;

  const StreamState *SS = State->get<StreamMap>(StreamSym);
  if (!SS)
    return;

  auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  assertStreamStateOpened(SS);

  State = State->set<StreamMap>(StreamSym,
                                StreamState::getOpened(Desc, ErrorNone, false));

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

  // FilePositionIndeterminate is not cleared.
  State = State->set<StreamMap>(
      StreamSym,
      StreamState::getOpened(Desc, ErrorNone, SS->FilePositionIndeterminate));
  C.addTransition(State);
}

void StreamChecker::evalFeofFerror(const FnDescription *Desc,
                                   const CallEvent &Call, CheckerContext &C,
                                   const StreamErrorState &ErrorKind) const {
  ProgramStateRef State = C.getState();
  SymbolRef StreamSym = getStreamArg(Desc, Call).getAsSymbol();
  if (!StreamSym)
    return;

  const CallExpr *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  const StreamState *SS = State->get<StreamMap>(StreamSym);
  if (!SS)
    return;

  assertStreamStateOpened(SS);

  if (SS->ErrorState & ErrorKind) {
    // Execution path with error of ErrorKind.
    // Function returns true.
    // From now on it is the only one error state.
    ProgramStateRef TrueState = bindAndAssumeTrue(State, C, CE);
    C.addTransition(TrueState->set<StreamMap>(
        StreamSym, StreamState::getOpened(Desc, ErrorKind,
                                          SS->FilePositionIndeterminate &&
                                              !ErrorKind.isFEof())));
  }
  if (StreamErrorState NewES = SS->ErrorState & (~ErrorKind)) {
    // Execution path(s) with ErrorKind not set.
    // Function returns false.
    // New error state is everything before minus ErrorKind.
    ProgramStateRef FalseState = bindInt(0, State, C, CE);
    C.addTransition(FalseState->set<StreamMap>(
        StreamSym,
        StreamState::getOpened(
            Desc, NewES, SS->FilePositionIndeterminate && !NewES.isFEof())));
  }
}

void StreamChecker::preDefault(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SVal StreamVal = getStreamArg(Desc, Call);
  State = ensureStreamNonNull(StreamVal, Call.getArgExpr(Desc->StreamArgNo), C,
                              State);
  if (!State)
    return;
  State = ensureStreamOpened(StreamVal, C, State);
  if (!State)
    return;

  C.addTransition(State);
}

void StreamChecker::evalSetFeofFerror(const FnDescription *Desc,
                                      const CallEvent &Call, CheckerContext &C,
                                      const StreamErrorState &ErrorKind) const {
  ProgramStateRef State = C.getState();
  SymbolRef StreamSym = getStreamArg(Desc, Call).getAsSymbol();
  assert(StreamSym && "Operation not permitted on non-symbolic stream value.");
  const StreamState *SS = State->get<StreamMap>(StreamSym);
  assert(SS && "Stream should be tracked by the checker.");
  State = State->set<StreamMap>(
      StreamSym, StreamState::getOpened(SS->LastOperation, ErrorKind));
  C.addTransition(State);
}

ProgramStateRef
StreamChecker::ensureStreamNonNull(SVal StreamVal, const Expr *StreamE,
                                   CheckerContext &C,
                                   ProgramStateRef State) const {
  auto Stream = StreamVal.getAs<DefinedSVal>();
  if (!Stream)
    return State;

  ConstraintManager &CM = C.getConstraintManager();

  ProgramStateRef StateNotNull, StateNull;
  std::tie(StateNotNull, StateNull) = CM.assumeDual(C.getState(), *Stream);

  if (!StateNotNull && StateNull) {
    if (ExplodedNode *N = C.generateErrorNode(StateNull)) {
      auto R = std::make_unique<PathSensitiveBugReport>(
          BT_FileNull, "Stream pointer might be NULL.", N);
      if (StreamE)
        bugreporter::trackExpressionValue(N, StreamE, *R);
      C.emitReport(std::move(R));
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
      C.emitReport(std::make_unique<PathSensitiveBugReport>(
          BT_UseAfterClose,
          "Stream might be already closed. Causes undefined behaviour.", N));
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
      C.emitReport(std::make_unique<PathSensitiveBugReport>(
          BT_UseAfterOpenFailed,
          "Stream might be invalid after "
          "(re-)opening it has failed. "
          "Can cause undefined behaviour.",
          N));
      return nullptr;
    }
    return State;
  }

  return State;
}

ProgramStateRef StreamChecker::ensureNoFilePositionIndeterminate(
    SVal StreamVal, CheckerContext &C, ProgramStateRef State) const {
  static const char *BugMessage =
      "File position of the stream might be 'indeterminate' "
      "after a failed operation. "
      "Can cause undefined behavior.";

  SymbolRef Sym = StreamVal.getAsSymbol();
  if (!Sym)
    return State;

  const StreamState *SS = State->get<StreamMap>(Sym);
  if (!SS)
    return State;

  assert(SS->isOpened() && "First ensure that stream is opened.");

  if (SS->FilePositionIndeterminate) {
    if (SS->ErrorState & ErrorFEof) {
      // The error is unknown but may be FEOF.
      // Continue analysis with the FEOF error state.
      // Report warning because the other possible error states.
      ExplodedNode *N = C.generateNonFatalErrorNode(State);
      if (!N)
        return nullptr;

      C.emitReport(std::make_unique<PathSensitiveBugReport>(
          BT_IndeterminatePosition, BugMessage, N));
      return State->set<StreamMap>(
          Sym, StreamState::getOpened(SS->LastOperation, ErrorFEof, false));
    }

    // Known or unknown error state without FEOF possible.
    // Stop analysis, report error.
    ExplodedNode *N = C.generateErrorNode(State);
    if (N)
      C.emitReport(std::make_unique<PathSensitiveBugReport>(
          BT_IndeterminatePosition, BugMessage, N));

    return nullptr;
  }

  return State;
}

ProgramStateRef
StreamChecker::ensureFseekWhenceCorrect(SVal WhenceVal, CheckerContext &C,
                                        ProgramStateRef State) const {
  std::optional<nonloc::ConcreteInt> CI =
      WhenceVal.getAs<nonloc::ConcreteInt>();
  if (!CI)
    return State;

  int64_t X = CI->getValue().getSExtValue();
  if (X == SeekSetVal || X == SeekCurVal || X == SeekEndVal)
    return State;

  if (ExplodedNode *N = C.generateNonFatalErrorNode(State)) {
    C.emitReport(std::make_unique<PathSensitiveBugReport>(
        BT_IllegalWhence,
        "The whence argument to fseek() should be "
        "SEEK_SET, SEEK_END, or SEEK_CUR.",
        N));
    return nullptr;
  }

  return State;
}

void StreamChecker::reportFEofWarning(SymbolRef StreamSym, CheckerContext &C,
                                      ProgramStateRef State) const {
  if (ExplodedNode *N = C.generateNonFatalErrorNode(State)) {
    auto R = std::make_unique<PathSensitiveBugReport>(
        BT_StreamEof,
        "Read function called when stream is in EOF state. "
        "Function has no effect.",
        N);
    R->markInteresting(StreamSym);
    C.emitReport(std::move(R));
    return;
  }
  C.addTransition(State);
}

ExplodedNode *
StreamChecker::reportLeaks(const SmallVector<SymbolRef, 2> &LeakedSyms,
                           CheckerContext &C, ExplodedNode *Pred) const {
  ExplodedNode *Err = C.generateNonFatalErrorNode(C.getState(), Pred);
  if (!Err)
    return Pred;

  for (SymbolRef LeakSym : LeakedSyms) {
    // Resource leaks can result in multiple warning that describe the same kind
    // of programming error:
    //  void f() {
    //    FILE *F = fopen("a.txt");
    //    if (rand()) // state split
    //      return; // warning
    //  } // warning
    // While this isn't necessarily true (leaking the same stream could result
    // from a different kinds of errors), the reduction in redundant reports
    // makes this a worthwhile heuristic.
    // FIXME: Add a checker option to turn this uniqueing feature off.
    const ExplodedNode *StreamOpenNode = getAcquisitionSite(Err, LeakSym, C);
    assert(StreamOpenNode && "Could not find place of stream opening.");

    PathDiagnosticLocation LocUsedForUniqueing;
    if (const Stmt *StreamStmt = StreamOpenNode->getStmtForDiagnostics())
       LocUsedForUniqueing = PathDiagnosticLocation::createBegin(
          StreamStmt, C.getSourceManager(),
          StreamOpenNode->getLocationContext());

    std::unique_ptr<PathSensitiveBugReport> R =
        std::make_unique<PathSensitiveBugReport>(
            BT_ResourceLeak,
            "Opened stream never closed. Potential resource leak.", Err,
            LocUsedForUniqueing,
            StreamOpenNode->getLocationContext()->getDecl());
    R->markInteresting(LeakSym);
    C.emitReport(std::move(R));
  }

  return Err;
}

void StreamChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  llvm::SmallVector<SymbolRef, 2> LeakedSyms;

  const StreamMapTy &Map = State->get<StreamMap>();
  for (const auto &I : Map) {
    SymbolRef Sym = I.first;
    const StreamState &SS = I.second;
    if (!SymReaper.isDead(Sym))
      continue;
    if (SS.isOpened())
      LeakedSyms.push_back(Sym);
    State = State->remove<StreamMap>(Sym);
  }

  ExplodedNode *N = C.getPredecessor();
  if (!LeakedSyms.empty())
    N = reportLeaks(LeakedSyms, C, N);

  C.addTransition(State, N);
}

ProgramStateRef StreamChecker::checkPointerEscape(
    ProgramStateRef State, const InvalidatedSymbols &Escaped,
    const CallEvent *Call, PointerEscapeKind Kind) const {
  // Check for file-handling system call that is not handled by the checker.
  // FIXME: The checker should be updated to handle all system calls that take
  // 'FILE*' argument. These are now ignored.
  if (Kind == PSK_DirectEscapeOnCall && Call->isInSystemHeader())
    return State;

  for (SymbolRef Sym : Escaped) {
    // The symbol escaped.
    // From now the stream can be manipulated in unknown way to the checker,
    // it is not possible to handle it any more.
    // Optimistically, assume that the corresponding file handle will be closed
    // somewhere else.
    // Remove symbol from state so the following stream calls on this symbol are
    // not handled by the checker.
    State = State->remove<StreamMap>(Sym);
  }
  return State;
}

//===----------------------------------------------------------------------===//
// Checker registration.
//===----------------------------------------------------------------------===//

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
