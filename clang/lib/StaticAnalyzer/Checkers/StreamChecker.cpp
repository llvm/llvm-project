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
#include "llvm/ADT/Sequence.h"
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

// This map holds the state of a stream.
// The stream is identified with a SymbolRef that is created when a stream
// opening function is modeled by the checker.
REGISTER_MAP_WITH_PROGRAMSTATE(StreamMap, SymbolRef, StreamState)

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

inline void assertStreamStateOpened(const StreamState *SS) {
  assert(SS->isOpened() && "Stream is expected to be opened");
}

struct StreamOperationEvaluator {
  SValBuilder &SVB;
  const ASTContext &ACtx;

  SymbolRef StreamSym;
  const StreamState *SS = nullptr;
  const CallExpr *CE = nullptr;

  StreamOperationEvaluator(CheckerContext &C)
      : SVB(C.getSValBuilder()), ACtx(C.getASTContext()) {
    ;
  }

  bool Init(const FnDescription *Desc, const CallEvent &Call, CheckerContext &C,
            ProgramStateRef State) {
    StreamSym = getStreamArg(Desc, Call).getAsSymbol();
    if (!StreamSym)
      return false;
    SS = State->get<StreamMap>(StreamSym);
    if (!SS)
      return false;
    CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
    if (!CE)
      return false;

    assertStreamStateOpened(SS);

    return true;
  }

  bool isStreamEof() const { return SS->ErrorState == ErrorFEof; }

  NonLoc getZeroVal(const CallEvent &Call) {
    return *SVB.makeZeroVal(Call.getResultType()).getAs<NonLoc>();
  }

  ProgramStateRef setStreamState(ProgramStateRef State,
                                 const StreamState &NewSS) {
    return State->set<StreamMap>(StreamSym, NewSS);
  }

  ProgramStateRef makeAndBindRetVal(ProgramStateRef State, CheckerContext &C) {
    NonLoc RetVal = makeRetVal(C, CE).castAs<NonLoc>();
    return State->BindExpr(CE, C.getLocationContext(), RetVal);
  }

  ProgramStateRef bindReturnValue(ProgramStateRef State, CheckerContext &C,
                                  uint64_t Val) {
    return State->BindExpr(CE, C.getLocationContext(),
                           SVB.makeIntVal(Val, CE->getCallReturnType(ACtx)));
  }

  ProgramStateRef bindReturnValue(ProgramStateRef State, CheckerContext &C,
                                  SVal Val) {
    return State->BindExpr(CE, C.getLocationContext(), Val);
  }

  ProgramStateRef bindNullReturnValue(ProgramStateRef State,
                                      CheckerContext &C) {
    return State->BindExpr(CE, C.getLocationContext(),
                           C.getSValBuilder().makeNullWithType(CE->getType()));
  }

  ProgramStateRef assumeBinOpNN(ProgramStateRef State,
                                BinaryOperator::Opcode Op, NonLoc LHS,
                                NonLoc RHS) {
    auto Cond = SVB.evalBinOpNN(State, Op, LHS, RHS, SVB.getConditionType())
                    .getAs<DefinedOrUnknownSVal>();
    if (!Cond)
      return nullptr;
    return State->assume(*Cond, true);
  }

  ConstraintManager::ProgramStatePair
  makeRetValAndAssumeDual(ProgramStateRef State, CheckerContext &C) {
    DefinedSVal RetVal = makeRetVal(C, CE);
    State = State->BindExpr(CE, C.getLocationContext(), RetVal);
    return C.getConstraintManager().assumeDual(State, RetVal);
  }
};

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
      {{{"fopen"}, 2}, {nullptr, &StreamChecker::evalFopen, ArgNone}},
      {{{"fdopen"}, 2}, {nullptr, &StreamChecker::evalFopen, ArgNone}},
      {{{"freopen"}, 3},
       {&StreamChecker::preFreopen, &StreamChecker::evalFreopen, 2}},
      {{{"tmpfile"}, 0}, {nullptr, &StreamChecker::evalFopen, ArgNone}},
      {{{"fclose"}, 1},
       {&StreamChecker::preDefault, &StreamChecker::evalFclose, 0}},
      {{{"fread"}, 4},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, true),
        std::bind(&StreamChecker::evalFreadFwrite, _1, _2, _3, _4, true), 3}},
      {{{"fwrite"}, 4},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, false),
        std::bind(&StreamChecker::evalFreadFwrite, _1, _2, _3, _4, false), 3}},
      {{{"fgetc"}, 1},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, true),
        std::bind(&StreamChecker::evalFgetx, _1, _2, _3, _4, true), 0}},
      {{{"fgets"}, 3},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, true),
        std::bind(&StreamChecker::evalFgetx, _1, _2, _3, _4, false), 2}},
      {{{"getc"}, 1},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, true),
        std::bind(&StreamChecker::evalFgetx, _1, _2, _3, _4, true), 0}},
      {{{"fputc"}, 2},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, false),
        std::bind(&StreamChecker::evalFputx, _1, _2, _3, _4, true), 1}},
      {{{"fputs"}, 2},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, false),
        std::bind(&StreamChecker::evalFputx, _1, _2, _3, _4, false), 1}},
      {{{"putc"}, 2},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, false),
        std::bind(&StreamChecker::evalFputx, _1, _2, _3, _4, true), 1}},
      {{{"fprintf"}},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, false),
        std::bind(&StreamChecker::evalFprintf, _1, _2, _3, _4), 0}},
      {{{"vfprintf"}, 3},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, false),
        std::bind(&StreamChecker::evalFprintf, _1, _2, _3, _4), 0}},
      {{{"fscanf"}},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, true),
        std::bind(&StreamChecker::evalFscanf, _1, _2, _3, _4), 0}},
      {{{"vfscanf"}, 3},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, true),
        std::bind(&StreamChecker::evalFscanf, _1, _2, _3, _4), 0}},
      {{{"ungetc"}, 2},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, false),
        std::bind(&StreamChecker::evalUngetc, _1, _2, _3, _4), 1}},
      {{{"getdelim"}, 4},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, true),
        std::bind(&StreamChecker::evalGetdelim, _1, _2, _3, _4), 3}},
      {{{"getline"}, 3},
       {std::bind(&StreamChecker::preReadWrite, _1, _2, _3, _4, true),
        std::bind(&StreamChecker::evalGetdelim, _1, _2, _3, _4), 2}},
      {{{"fseek"}, 3},
       {&StreamChecker::preFseek, &StreamChecker::evalFseek, 0}},
      {{{"fseeko"}, 3},
       {&StreamChecker::preFseek, &StreamChecker::evalFseek, 0}},
      {{{"ftell"}, 1},
       {&StreamChecker::preDefault, &StreamChecker::evalFtell, 0}},
      {{{"ftello"}, 1},
       {&StreamChecker::preDefault, &StreamChecker::evalFtell, 0}},
      {{{"fflush"}, 1},
       {&StreamChecker::preFflush, &StreamChecker::evalFflush, 0}},
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
      {{{"fileno"}, 1},
       {&StreamChecker::preDefault, &StreamChecker::evalFileno, 0}},
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

  void preReadWrite(const FnDescription *Desc, const CallEvent &Call,
                    CheckerContext &C, bool IsRead) const;

  void evalFreadFwrite(const FnDescription *Desc, const CallEvent &Call,
                       CheckerContext &C, bool IsFread) const;

  void evalFgetx(const FnDescription *Desc, const CallEvent &Call,
                 CheckerContext &C, bool SingleChar) const;

  void evalFputx(const FnDescription *Desc, const CallEvent &Call,
                 CheckerContext &C, bool IsSingleChar) const;

  void evalFprintf(const FnDescription *Desc, const CallEvent &Call,
                   CheckerContext &C) const;

  void evalFscanf(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;

  void evalUngetc(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;

  void evalGetdelim(const FnDescription *Desc, const CallEvent &Call,
                    CheckerContext &C) const;

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

  void preFflush(const FnDescription *Desc, const CallEvent &Call,
                 CheckerContext &C) const;

  void evalFflush(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;

  void evalFileno(const FnDescription *Desc, const CallEvent &Call,
                  CheckerContext &C) const;

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
  const NoteTag *constructNoteTag(CheckerContext &C, SymbolRef StreamSym,
                                  const std::string &Message) const {
    return C.getNoteTag([this, StreamSym,
                         Message](PathSensitiveBugReport &BR) -> std::string {
      if (BR.isInteresting(StreamSym) && &BR.getBugType() == &BT_ResourceLeak)
        return Message;
      return "";
    });
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

static ProgramStateRef escapeArgs(ProgramStateRef State, CheckerContext &C,
                                  const CallEvent &Call,
                                  ArrayRef<unsigned int> EscapingArgs) {
  const auto *CE = Call.getOriginExpr();

  SmallVector<SVal> EscapingVals;
  EscapingVals.reserve(EscapingArgs.size());
  for (auto EscArgIdx : EscapingArgs)
    EscapingVals.push_back(Call.getArgSVal(EscArgIdx));
  State = State->invalidateRegions(EscapingVals, CE, C.blockCount(),
                                   C.getLocationContext(),
                                   /*CausesPointerEscape=*/false);
  return State;
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
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  // Close the File Descriptor.
  // Regardless if the close fails or not, stream becomes "closed"
  // and can not be used any more.
  State = E.setStreamState(State, StreamState::getClosed(Desc));

  // Return 0 on success, EOF on failure.
  C.addTransition(E.bindReturnValue(State, C, 0));
  C.addTransition(E.bindReturnValue(State, C, *EofVal));
}

void StreamChecker::preReadWrite(const FnDescription *Desc,
                                 const CallEvent &Call, CheckerContext &C,
                                 bool IsRead) const {
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

  if (!IsRead) {
    C.addTransition(State);
    return;
  }

  SymbolRef Sym = StreamVal.getAsSymbol();
  if (Sym && State->get<StreamMap>(Sym)) {
    const StreamState *SS = State->get<StreamMap>(Sym);
    if (SS->ErrorState & ErrorFEof)
      reportFEofWarning(Sym, C, State);
  } else {
    C.addTransition(State);
  }
}

void StreamChecker::evalFreadFwrite(const FnDescription *Desc,
                                    const CallEvent &Call, CheckerContext &C,
                                    bool IsFread) const {
  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  std::optional<NonLoc> SizeVal = Call.getArgSVal(1).getAs<NonLoc>();
  if (!SizeVal)
    return;
  std::optional<NonLoc> NMembVal = Call.getArgSVal(2).getAs<NonLoc>();
  if (!NMembVal)
    return;

  // C'99 standard, §7.19.8.1.3, the return value of fread:
  // The fread function returns the number of elements successfully read, which
  // may be less than nmemb if a read error or end-of-file is encountered. If
  // size or nmemb is zero, fread returns zero and the contents of the array and
  // the state of the stream remain unchanged.
  if (State->isNull(*SizeVal).isConstrainedTrue() ||
      State->isNull(*NMembVal).isConstrainedTrue()) {
    // This is the "size or nmemb is zero" case.
    // Just return 0, do nothing more (not clear the error flags).
    C.addTransition(E.bindReturnValue(State, C, 0));
    return;
  }

  // At read, invalidate the buffer in any case of error or success,
  // except if EOF was already present.
  if (IsFread && !E.isStreamEof())
    State = escapeArgs(State, C, Call, {0});

  // Generate a transition for the success state.
  // If we know the state to be FEOF at fread, do not add a success state.
  if (!IsFread || !E.isStreamEof()) {
    ProgramStateRef StateNotFailed =
        State->BindExpr(E.CE, C.getLocationContext(), *NMembVal);
    StateNotFailed =
        E.setStreamState(StateNotFailed, StreamState::getOpened(Desc));
    C.addTransition(StateNotFailed);
  }

  // Add transition for the failed state.
  NonLoc RetVal = makeRetVal(C, E.CE).castAs<NonLoc>();
  ProgramStateRef StateFailed =
      State->BindExpr(E.CE, C.getLocationContext(), RetVal);
  StateFailed = E.assumeBinOpNN(StateFailed, BO_LT, RetVal, *NMembVal);
  if (!StateFailed)
    return;

  StreamErrorState NewES;
  if (IsFread)
    NewES = E.isStreamEof() ? ErrorFEof : ErrorFEof | ErrorFError;
  else
    NewES = ErrorFError;
  // If a (non-EOF) error occurs, the resulting value of the file position
  // indicator for the stream is indeterminate.
  StateFailed = E.setStreamState(
      StateFailed, StreamState::getOpened(Desc, NewES, !NewES.isFEof()));
  if (IsFread && !E.isStreamEof())
    C.addTransition(StateFailed, constructSetEofNoteTag(C, E.StreamSym));
  else
    C.addTransition(StateFailed);
}

void StreamChecker::evalFgetx(const FnDescription *Desc, const CallEvent &Call,
                              CheckerContext &C, bool SingleChar) const {
  // `fgetc` returns the read character on success, otherwise returns EOF.
  // `fgets` returns the read buffer address on success, otherwise returns NULL.

  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  if (!E.isStreamEof()) {
    // If there was already EOF, assume that read buffer is not changed.
    // Otherwise it may change at success or failure.
    State = escapeArgs(State, C, Call, {0});
    if (SingleChar) {
      // Generate a transition for the success state of `fgetc`.
      NonLoc RetVal = makeRetVal(C, E.CE).castAs<NonLoc>();
      ProgramStateRef StateNotFailed =
          State->BindExpr(E.CE, C.getLocationContext(), RetVal);
      // The returned 'unsigned char' of `fgetc` is converted to 'int',
      // so we need to check if it is in range [0, 255].
      StateNotFailed = StateNotFailed->assumeInclusiveRange(
          RetVal,
          E.SVB.getBasicValueFactory().getValue(0, E.ACtx.UnsignedCharTy),
          E.SVB.getBasicValueFactory().getMaxValue(E.ACtx.UnsignedCharTy),
          true);
      if (!StateNotFailed)
        return;
      C.addTransition(StateNotFailed);
    } else {
      // Generate a transition for the success state of `fgets`.
      std::optional<DefinedSVal> GetBuf =
          Call.getArgSVal(0).getAs<DefinedSVal>();
      if (!GetBuf)
        return;
      ProgramStateRef StateNotFailed =
          State->BindExpr(E.CE, C.getLocationContext(), *GetBuf);
      StateNotFailed =
          E.setStreamState(StateNotFailed, StreamState::getOpened(Desc));
      C.addTransition(StateNotFailed);
    }
  }

  // Add transition for the failed state.
  ProgramStateRef StateFailed;
  if (SingleChar)
    StateFailed = E.bindReturnValue(State, C, *EofVal);
  else
    StateFailed = E.bindNullReturnValue(State, C);

  // If a (non-EOF) error occurs, the resulting value of the file position
  // indicator for the stream is indeterminate.
  StreamErrorState NewES =
      E.isStreamEof() ? ErrorFEof : ErrorFEof | ErrorFError;
  StateFailed = E.setStreamState(
      StateFailed, StreamState::getOpened(Desc, NewES, !NewES.isFEof()));
  if (!E.isStreamEof())
    C.addTransition(StateFailed, constructSetEofNoteTag(C, E.StreamSym));
  else
    C.addTransition(StateFailed);
}

void StreamChecker::evalFputx(const FnDescription *Desc, const CallEvent &Call,
                              CheckerContext &C, bool IsSingleChar) const {
  // `fputc` returns the written character on success, otherwise returns EOF.
  // `fputs` returns a nonnegative value on success, otherwise returns EOF.

  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  if (IsSingleChar) {
    // Generate a transition for the success state of `fputc`.
    std::optional<NonLoc> PutVal = Call.getArgSVal(0).getAs<NonLoc>();
    if (!PutVal)
      return;
    ProgramStateRef StateNotFailed =
        State->BindExpr(E.CE, C.getLocationContext(), *PutVal);
    StateNotFailed =
        E.setStreamState(StateNotFailed, StreamState::getOpened(Desc));
    C.addTransition(StateNotFailed);
  } else {
    // Generate a transition for the success state of `fputs`.
    NonLoc RetVal = makeRetVal(C, E.CE).castAs<NonLoc>();
    ProgramStateRef StateNotFailed =
        State->BindExpr(E.CE, C.getLocationContext(), RetVal);
    StateNotFailed =
        E.assumeBinOpNN(StateNotFailed, BO_GE, RetVal, E.getZeroVal(Call));
    if (!StateNotFailed)
      return;
    StateNotFailed =
        E.setStreamState(StateNotFailed, StreamState::getOpened(Desc));
    C.addTransition(StateNotFailed);
  }

  // Add transition for the failed state. The resulting value of the file
  // position indicator for the stream is indeterminate.
  ProgramStateRef StateFailed = E.bindReturnValue(State, C, *EofVal);
  StateFailed = E.setStreamState(
      StateFailed, StreamState::getOpened(Desc, ErrorFError, true));
  C.addTransition(StateFailed);
}

void StreamChecker::evalFprintf(const FnDescription *Desc,
                                const CallEvent &Call,
                                CheckerContext &C) const {
  if (Call.getNumArgs() < 2)
    return;

  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  NonLoc RetVal = makeRetVal(C, E.CE).castAs<NonLoc>();
  State = State->BindExpr(E.CE, C.getLocationContext(), RetVal);
  auto Cond =
      E.SVB
          .evalBinOp(State, BO_GE, RetVal, E.SVB.makeZeroVal(E.ACtx.IntTy),
                     E.SVB.getConditionType())
          .getAs<DefinedOrUnknownSVal>();
  if (!Cond)
    return;
  ProgramStateRef StateNotFailed, StateFailed;
  std::tie(StateNotFailed, StateFailed) = State->assume(*Cond);

  StateNotFailed =
      E.setStreamState(StateNotFailed, StreamState::getOpened(Desc));
  C.addTransition(StateNotFailed);

  // Add transition for the failed state. The resulting value of the file
  // position indicator for the stream is indeterminate.
  StateFailed = E.setStreamState(
      StateFailed, StreamState::getOpened(Desc, ErrorFError, true));
  C.addTransition(StateFailed);
}

void StreamChecker::evalFscanf(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  if (Call.getNumArgs() < 2)
    return;

  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  // Add the success state.
  // In this context "success" means there is not an EOF or other read error
  // before any item is matched in 'fscanf'. But there may be match failure,
  // therefore return value can be 0 or greater.
  // It is not specified what happens if some items (not all) are matched and
  // then EOF or read error happens. Now this case is handled like a "success"
  // case, and no error flags are set on the stream. This is probably not
  // accurate, and the POSIX documentation does not tell more.
  if (!E.isStreamEof()) {
    NonLoc RetVal = makeRetVal(C, E.CE).castAs<NonLoc>();
    ProgramStateRef StateNotFailed =
        State->BindExpr(E.CE, C.getLocationContext(), RetVal);
    StateNotFailed =
        E.assumeBinOpNN(StateNotFailed, BO_GE, RetVal, E.getZeroVal(Call));
    if (!StateNotFailed)
      return;

    if (auto const *Callee = Call.getCalleeIdentifier();
        !Callee || !Callee->getName().equals("vfscanf")) {
      SmallVector<unsigned int> EscArgs;
      for (auto EscArg : llvm::seq(2u, Call.getNumArgs()))
        EscArgs.push_back(EscArg);
      StateNotFailed = escapeArgs(StateNotFailed, C, Call, EscArgs);
    }

    if (StateNotFailed)
      C.addTransition(StateNotFailed);
  }

  // Add transition for the failed state.
  // Error occurs if nothing is matched yet and reading the input fails.
  // Error can be EOF, or other error. At "other error" FERROR or 'errno' can
  // be set but it is not further specified if all are required to be set.
  // Documentation does not mention, but file position will be set to
  // indeterminate similarly as at 'fread'.
  ProgramStateRef StateFailed = E.bindReturnValue(State, C, *EofVal);
  StreamErrorState NewES =
      E.isStreamEof() ? ErrorFEof : ErrorNone | ErrorFEof | ErrorFError;
  StateFailed = E.setStreamState(
      StateFailed, StreamState::getOpened(Desc, NewES, !NewES.isFEof()));
  if (!E.isStreamEof())
    C.addTransition(StateFailed, constructSetEofNoteTag(C, E.StreamSym));
  else
    C.addTransition(StateFailed);
}

void StreamChecker::evalUngetc(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  // Generate a transition for the success state.
  std::optional<NonLoc> PutVal = Call.getArgSVal(0).getAs<NonLoc>();
  if (!PutVal)
    return;
  ProgramStateRef StateNotFailed = E.bindReturnValue(State, C, *PutVal);
  StateNotFailed =
      E.setStreamState(StateNotFailed, StreamState::getOpened(Desc));
  C.addTransition(StateNotFailed);

  // Add transition for the failed state.
  // Failure of 'ungetc' does not result in feof or ferror state.
  // If the PutVal has value of EofVal the function should "fail", but this is
  // the same transition as the success state.
  // In this case only one state transition is added by the analyzer (the two
  // new states may be similar).
  ProgramStateRef StateFailed = E.bindReturnValue(State, C, *EofVal);
  StateFailed = E.setStreamState(StateFailed, StreamState::getOpened(Desc));
  C.addTransition(StateFailed);
}

void StreamChecker::evalGetdelim(const FnDescription *Desc,
                                 const CallEvent &Call,
                                 CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  // Upon successful completion, the getline() and getdelim() functions shall
  // return the number of bytes written into the buffer.
  // If the end-of-file indicator for the stream is set, the function shall
  // return -1.
  // If an error occurs, the function shall return -1 and set 'errno'.

  if (!E.isStreamEof()) {
    // Escape buffer and size (may change by the call).
    // May happen even at error (partial read?).
    State = escapeArgs(State, C, Call, {0, 1});

    // Add transition for the successful state.
    NonLoc RetVal = makeRetVal(C, E.CE).castAs<NonLoc>();
    ProgramStateRef StateNotFailed =
        State->BindExpr(E.CE, C.getLocationContext(), RetVal);
    StateNotFailed =
        E.assumeBinOpNN(StateNotFailed, BO_GE, RetVal, E.getZeroVal(Call));
    if (!StateNotFailed)
      return;
    C.addTransition(StateNotFailed);
  }

  // Add transition for the failed state.
  // If a (non-EOF) error occurs, the resulting value of the file position
  // indicator for the stream is indeterminate.
  ProgramStateRef StateFailed = E.bindReturnValue(State, C, -1);
  StreamErrorState NewES =
      E.isStreamEof() ? ErrorFEof : ErrorFEof | ErrorFError;
  StateFailed = E.setStreamState(
      StateFailed, StreamState::getOpened(Desc, NewES, !NewES.isFEof()));
  if (E.isStreamEof())
    C.addTransition(StateFailed, constructSetEofNoteTag(C, E.StreamSym));
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
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  const llvm::APSInt *PosV =
      C.getSValBuilder().getKnownValue(State, Call.getArgSVal(1));
  const llvm::APSInt *WhenceV =
      C.getSValBuilder().getKnownValue(State, Call.getArgSVal(2));

  // Bifurcate the state into failed and non-failed.
  // Return zero on success, nonzero on error.
  ProgramStateRef StateNotFailed, StateFailed;
  std::tie(StateFailed, StateNotFailed) = E.makeRetValAndAssumeDual(State, C);

  // No failure: Reset the state to opened with no error.
  StateNotFailed =
      E.setStreamState(StateNotFailed, StreamState::getOpened(Desc));
  C.addTransition(StateNotFailed);

  // At error it is possible that fseek fails but sets none of the error flags.
  // If fseek failed, assume that the file position becomes indeterminate in any
  // case.
  StreamErrorState NewErrS = ErrorNone | ErrorFError;
  // Setting the position to start of file never produces EOF error.
  if (!(PosV && *PosV == 0 && WhenceV && *WhenceV == SeekSetVal))
    NewErrS = NewErrS | ErrorFEof;
  StateFailed = E.setStreamState(StateFailed,
                                 StreamState::getOpened(Desc, NewErrS, true));
  C.addTransition(StateFailed, constructSetEofNoteTag(C, E.StreamSym));
}

void StreamChecker::evalFgetpos(const FnDescription *Desc,
                                const CallEvent &Call,
                                CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  ProgramStateRef StateNotFailed, StateFailed;
  std::tie(StateFailed, StateNotFailed) = E.makeRetValAndAssumeDual(State, C);
  StateNotFailed = escapeArgs(StateNotFailed, C, Call, {1});

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
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  ProgramStateRef StateNotFailed, StateFailed;
  std::tie(StateFailed, StateNotFailed) = E.makeRetValAndAssumeDual(State, C);

  StateNotFailed = E.setStreamState(
      StateNotFailed, StreamState::getOpened(Desc, ErrorNone, false));

  // At failure ferror could be set.
  // The standards do not tell what happens with the file position at failure.
  // But we can assume that it is dangerous to make a next I/O operation after
  // the position was not set correctly (similar to 'fseek').
  StateFailed = E.setStreamState(
      StateFailed, StreamState::getOpened(Desc, ErrorNone | ErrorFError, true));

  C.addTransition(StateNotFailed);
  C.addTransition(StateFailed);
}

void StreamChecker::evalFtell(const FnDescription *Desc, const CallEvent &Call,
                              CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  NonLoc RetVal = makeRetVal(C, E.CE).castAs<NonLoc>();
  ProgramStateRef StateNotFailed =
      State->BindExpr(E.CE, C.getLocationContext(), RetVal);
  StateNotFailed =
      E.assumeBinOpNN(StateNotFailed, BO_GE, RetVal, E.getZeroVal(Call));
  if (!StateNotFailed)
    return;

  ProgramStateRef StateFailed = E.bindReturnValue(State, C, -1);

  // This function does not affect the stream state.
  // Still we add success and failure state with the appropriate return value.
  // StdLibraryFunctionsChecker can change these states (set the 'errno' state).
  C.addTransition(StateNotFailed);
  C.addTransition(StateFailed);
}

void StreamChecker::evalRewind(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  State =
      E.setStreamState(State, StreamState::getOpened(Desc, ErrorNone, false));
  C.addTransition(State);
}

void StreamChecker::preFflush(const FnDescription *Desc, const CallEvent &Call,
                              CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SVal StreamVal = getStreamArg(Desc, Call);
  std::optional<DefinedSVal> Stream = StreamVal.getAs<DefinedSVal>();
  if (!Stream)
    return;

  ProgramStateRef StateNotNull, StateNull;
  std::tie(StateNotNull, StateNull) =
      C.getConstraintManager().assumeDual(State, *Stream);
  if (StateNotNull && !StateNull)
    ensureStreamOpened(StreamVal, C, StateNotNull);
}

void StreamChecker::evalFflush(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SVal StreamVal = getStreamArg(Desc, Call);
  std::optional<DefinedSVal> Stream = StreamVal.getAs<DefinedSVal>();
  if (!Stream)
    return;

  // Skip if the stream can be both NULL and non-NULL.
  ProgramStateRef StateNotNull, StateNull;
  std::tie(StateNotNull, StateNull) =
      C.getConstraintManager().assumeDual(State, *Stream);
  if (StateNotNull && StateNull)
    return;
  if (StateNotNull && !StateNull)
    State = StateNotNull;
  else
    State = StateNull;

  const CallExpr *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return;

  // `fflush` returns EOF on failure, otherwise returns 0.
  ProgramStateRef StateFailed = bindInt(*EofVal, State, C, CE);
  ProgramStateRef StateNotFailed = bindInt(0, State, C, CE);

  // Clear error states if `fflush` returns 0, but retain their EOF flags.
  auto ClearErrorInNotFailed = [&StateNotFailed, Desc](SymbolRef Sym,
                                                       const StreamState *SS) {
    if (SS->ErrorState & ErrorFError) {
      StreamErrorState NewES =
          (SS->ErrorState & ErrorFEof) ? ErrorFEof : ErrorNone;
      StreamState NewSS = StreamState::getOpened(Desc, NewES, false);
      StateNotFailed = StateNotFailed->set<StreamMap>(Sym, NewSS);
    }
  };

  if (StateNotNull && !StateNull) {
    // Skip if the input stream's state is unknown, open-failed or closed.
    if (SymbolRef StreamSym = StreamVal.getAsSymbol()) {
      const StreamState *SS = State->get<StreamMap>(StreamSym);
      if (SS) {
        assert(SS->isOpened() && "Stream is expected to be opened");
        ClearErrorInNotFailed(StreamSym, SS);
      } else
        return;
    }
  } else {
    // Clear error states for all streams.
    const StreamMapTy &Map = StateNotFailed->get<StreamMap>();
    for (const auto &I : Map) {
      SymbolRef Sym = I.first;
      const StreamState &SS = I.second;
      if (SS.isOpened())
        ClearErrorInNotFailed(Sym, &SS);
    }
  }

  C.addTransition(StateNotFailed);
  C.addTransition(StateFailed);
}

void StreamChecker::evalClearerr(const FnDescription *Desc,
                                 const CallEvent &Call,
                                 CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  // FilePositionIndeterminate is not cleared.
  State = E.setStreamState(
      State,
      StreamState::getOpened(Desc, ErrorNone, E.SS->FilePositionIndeterminate));
  C.addTransition(State);
}

void StreamChecker::evalFeofFerror(const FnDescription *Desc,
                                   const CallEvent &Call, CheckerContext &C,
                                   const StreamErrorState &ErrorKind) const {
  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  if (E.SS->ErrorState & ErrorKind) {
    // Execution path with error of ErrorKind.
    // Function returns true.
    // From now on it is the only one error state.
    ProgramStateRef TrueState = bindAndAssumeTrue(State, C, E.CE);
    C.addTransition(E.setStreamState(
        TrueState, StreamState::getOpened(Desc, ErrorKind,
                                          E.SS->FilePositionIndeterminate &&
                                              !ErrorKind.isFEof())));
  }
  if (StreamErrorState NewES = E.SS->ErrorState & (~ErrorKind)) {
    // Execution path(s) with ErrorKind not set.
    // Function returns false.
    // New error state is everything before minus ErrorKind.
    ProgramStateRef FalseState = E.bindReturnValue(State, C, 0);
    C.addTransition(E.setStreamState(
        FalseState,
        StreamState::getOpened(
            Desc, NewES, E.SS->FilePositionIndeterminate && !NewES.isFEof())));
  }
}

void StreamChecker::evalFileno(const FnDescription *Desc, const CallEvent &Call,
                               CheckerContext &C) const {
  // Fileno should fail only if the passed pointer is invalid.
  // Some of the preconditions are checked already in preDefault.
  // Here we can assume that the operation does not fail, because if we
  // introduced a separate branch where fileno() returns -1, then it would cause
  // many unexpected and unwanted warnings in situations where fileno() is
  // called on valid streams.
  // The stream error states are not modified by 'fileno', and 'errno' is also
  // left unchanged (so this evalCall does not invalidate it, but we have a
  // custom evalCall instead of the default that would invalidate it).
  ProgramStateRef State = C.getState();
  StreamOperationEvaluator E(C);
  if (!E.Init(Desc, Call, C, State))
    return;

  NonLoc RetVal = makeRetVal(C, E.CE).castAs<NonLoc>();
  State = State->BindExpr(E.CE, C.getLocationContext(), RetVal);
  State = E.assumeBinOpNN(State, BO_GE, RetVal, E.getZeroVal(Call));
  if (!State)
    return;

  C.addTransition(State);
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
  std::tie(StateNotNull, StateNull) = CM.assumeDual(State, *Stream);

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
