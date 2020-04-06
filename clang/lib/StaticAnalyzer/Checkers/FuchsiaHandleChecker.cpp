//=== FuchsiaHandleChecker.cpp - Find handle leaks/double closes -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker checks if the handle of Fuchsia is properly used according to
// following rules.
//   - If a handle is acquired, it should be released before execution
//        ends.
//   - If a handle is released, it should not be released again.
//   - If a handle is released, it should not be used for other purposes
//        such as I/O.
//
// In this checker, each tracked handle is associated with a state. When the
// handle variable is passed to different function calls or syscalls, its state
// changes. The state changes can be generally represented by following ASCII
// Art:
//
//
//                              +-+---------v-+         +------------+
//       acquire_func succeeded |             | Escape  |            |
//            +----------------->  Allocated  +--------->  Escaped   <--+
//            |                 |             |         |            |  |
//            |                 +-----+------++         +------------+  |
//            |                       |      |                          |
//            |         release_func  |      +--+                       |
//            |                       |         | handle  +--------+    |
//            |                       |         | dies    |        |    |
//            |                  +----v-----+   +---------> Leaked |    |
//            |                  |          |             |(REPORT)|    |
// +----------+--+               | Released | Escape      +--------+    |
// |             |               |          +---------------------------+
// | Not tracked <--+            +----+---+-+
// |             |  |                 |   |        As argument by value
// +------+------+  |    release_func |   +------+ in function call
//        |         |                 |          | or by reference in
//        |         |                 |          | use_func call
//        +---------+            +----v-----+    |     +-----------+
//        acquire_func failed    | Double   |    +-----> Use after |
//                               | released |          | released  |
//                               | (REPORT) |          | (REPORT)  |
//                               +----------+          +-----------+
//
// acquire_func represents the functions or syscalls that may acquire a handle.
// release_func represents the functions or syscalls that may release a handle.
// use_func represents the functions or syscall that requires an open handle.
//
// If a tracked handle dies in "Released" or "Not Tracked" state, we assume it
// is properly used. Otherwise a bug and will be reported.
//
// Note that, the analyzer does not always know for sure if a function failed
// or succeeded. In those cases we use the state MaybeAllocated.
// Thus, the diagramm above captures the intent, not implementation details.
//
// Due to the fact that the number of handle related syscalls in Fuchsia
// is large, we adopt the annotation attributes to descript syscalls'
// operations(acquire/release/use) on handles instead of hardcoding
// everything in the checker.
//
// We use following annotation attributes for handle related syscalls or
// functions:
//  1. __attribute__((acquire_handle("Fuchsia"))) |handle will be acquired
//  2. __attribute__((release_handle("Fuchsia"))) |handle will be released
//  3. __attribute__((use_handle("Fuchsia"))) |handle will not transit to
//     escaped state, it also needs to be open.
//
// For example, an annotated syscall:
//   zx_status_t zx_channel_create(
//   uint32_t options,
//   zx_handle_t* out0 __attribute__((acquire_handle("Fuchsia"))) ,
//   zx_handle_t* out1 __attribute__((acquire_handle("Fuchsia"))));
// denotes a syscall which will acquire two handles and save them to 'out0' and
// 'out1' when succeeded.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymExpr.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace ento;

namespace {

static const StringRef HandleTypeName = "zx_handle_t";
static const StringRef ErrorTypeName = "zx_status_t";

class HandleState {
private:
  enum class Kind { MaybeAllocated, Allocated, Released, Escaped } K;
  SymbolRef ErrorSym;
  HandleState(Kind K, SymbolRef ErrorSym) : K(K), ErrorSym(ErrorSym) {}

public:
  bool operator==(const HandleState &Other) const {
    return K == Other.K && ErrorSym == Other.ErrorSym;
  }
  bool isAllocated() const { return K == Kind::Allocated; }
  bool maybeAllocated() const { return K == Kind::MaybeAllocated; }
  bool isReleased() const { return K == Kind::Released; }
  bool isEscaped() const { return K == Kind::Escaped; }

  static HandleState getMaybeAllocated(SymbolRef ErrorSym) {
    return HandleState(Kind::MaybeAllocated, ErrorSym);
  }
  static HandleState getAllocated(ProgramStateRef State, HandleState S) {
    assert(S.maybeAllocated());
    assert(State->getConstraintManager()
               .isNull(State, S.getErrorSym())
               .isConstrained());
    return HandleState(Kind::Allocated, nullptr);
  }
  static HandleState getReleased() {
    return HandleState(Kind::Released, nullptr);
  }
  static HandleState getEscaped() {
    return HandleState(Kind::Escaped, nullptr);
  }

  SymbolRef getErrorSym() const { return ErrorSym; }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(static_cast<int>(K));
    ID.AddPointer(ErrorSym);
  }

  LLVM_DUMP_METHOD void dump(raw_ostream &OS) const {
    switch (K) {
#define CASE(ID)                                                               \
  case ID:                                                                     \
    OS << #ID;                                                                 \
    break;
      CASE(Kind::MaybeAllocated)
      CASE(Kind::Allocated)
      CASE(Kind::Released)
      CASE(Kind::Escaped)
    }
    if (ErrorSym) {
      OS << " ErrorSym: ";
      ErrorSym->dumpToStream(OS);
    }
  }

  LLVM_DUMP_METHOD void dump() const { dump(llvm::errs()); }
};

template <typename Attr> static bool hasFuchsiaAttr(const Decl *D) {
  return D->hasAttr<Attr>() && D->getAttr<Attr>()->getHandleType() == "Fuchsia";
}

class FuchsiaHandleChecker
    : public Checker<check::PostCall, check::PreCall, check::DeadSymbols,
                     check::PointerEscape, eval::Assume> {
  BugType LeakBugType{this, "Fuchsia handle leak", "Fuchsia Handle Error",
                      /*SuppressOnSink=*/true};
  BugType DoubleReleaseBugType{this, "Fuchsia handle double release",
                               "Fuchsia Handle Error"};
  BugType UseAfterReleaseBugType{this, "Fuchsia handle use after release",
                                 "Fuchsia Handle Error"};

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  ProgramStateRef evalAssume(ProgramStateRef State, SVal Cond,
                             bool Assumption) const;
  ProgramStateRef checkPointerEscape(ProgramStateRef State,
                                     const InvalidatedSymbols &Escaped,
                                     const CallEvent *Call,
                                     PointerEscapeKind Kind) const;

  ExplodedNode *reportLeaks(ArrayRef<SymbolRef> LeakedHandles,
                            CheckerContext &C, ExplodedNode *Pred) const;

  void reportDoubleRelease(SymbolRef HandleSym, const SourceRange &Range,
                           CheckerContext &C) const;

  void reportUseAfterFree(SymbolRef HandleSym, const SourceRange &Range,
                          CheckerContext &C) const;

  void reportBug(SymbolRef Sym, ExplodedNode *ErrorNode, CheckerContext &C,
                 const SourceRange *Range, const BugType &Type,
                 StringRef Msg) const;

  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;
};
} // end anonymous namespace

REGISTER_MAP_WITH_PROGRAMSTATE(HStateMap, SymbolRef, HandleState)

static const ExplodedNode *getAcquireSite(const ExplodedNode *N, SymbolRef Sym,
                                          CheckerContext &Ctx) {
  ProgramStateRef State = N->getState();
  // When bug type is handle leak, exploded node N does not have state info for
  // leaking handle. Get the predecessor of N instead.
  if (!State->get<HStateMap>(Sym))
    N = N->getFirstPred();

  const ExplodedNode *Pred = N;
  while (N) {
    State = N->getState();
    if (!State->get<HStateMap>(Sym)) {
      const HandleState *HState = Pred->getState()->get<HStateMap>(Sym);
      if (HState && (HState->isAllocated() || HState->maybeAllocated()))
        return N;
    }
    Pred = N;
    N = N->getFirstPred();
  }
  return nullptr;
}

/// Returns the symbols extracted from the argument or null if it cannot be
/// found.
static SymbolRef getFuchsiaHandleSymbol(QualType QT, SVal Arg,
                                        ProgramStateRef State) {
  int PtrToHandleLevel = 0;
  while (QT->isAnyPointerType() || QT->isReferenceType()) {
    ++PtrToHandleLevel;
    QT = QT->getPointeeType();
  }
  if (const auto *HandleType = QT->getAs<TypedefType>()) {
    if (HandleType->getDecl()->getName() != HandleTypeName)
      return nullptr;
    if (PtrToHandleLevel > 1) {
      // Not supported yet.
      return nullptr;
    }

    if (PtrToHandleLevel == 0) {
      return Arg.getAsSymbol();
    } else {
      assert(PtrToHandleLevel == 1);
      if (Optional<Loc> ArgLoc = Arg.getAs<Loc>())
        return State->getSVal(*ArgLoc).getAsSymbol();
    }
  }
  return nullptr;
}

void FuchsiaHandleChecker::checkPreCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const FunctionDecl *FuncDecl = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FuncDecl) {
    // Unknown call, escape by value handles. They are not covered by
    // PointerEscape callback.
    for (unsigned Arg = 0; Arg < Call.getNumArgs(); ++Arg) {
      if (SymbolRef Handle = Call.getArgSVal(Arg).getAsSymbol())
        State = State->set<HStateMap>(Handle, HandleState::getEscaped());
    }
    C.addTransition(State);
    return;
  }

  for (unsigned Arg = 0; Arg < Call.getNumArgs(); ++Arg) {
    if (Arg >= FuncDecl->getNumParams())
      break;
    const ParmVarDecl *PVD = FuncDecl->getParamDecl(Arg);
    SymbolRef Handle =
        getFuchsiaHandleSymbol(PVD->getType(), Call.getArgSVal(Arg), State);
    if (!Handle)
      continue;

    // Handled in checkPostCall.
    if (hasFuchsiaAttr<ReleaseHandleAttr>(PVD) ||
        hasFuchsiaAttr<AcquireHandleAttr>(PVD))
      continue;

    const HandleState *HState = State->get<HStateMap>(Handle);
    if (!HState || HState->isEscaped())
      continue;

    if (hasFuchsiaAttr<UseHandleAttr>(PVD) || PVD->getType()->isIntegerType()) {
      if (HState->isReleased()) {
        reportUseAfterFree(Handle, Call.getArgSourceRange(Arg), C);
        return;
      }
    }
    if (!hasFuchsiaAttr<UseHandleAttr>(PVD) &&
        PVD->getType()->isIntegerType()) {
      // Working around integer by-value escapes.
      State = State->set<HStateMap>(Handle, HandleState::getEscaped());
    }
  }
  C.addTransition(State);
}

void FuchsiaHandleChecker::checkPostCall(const CallEvent &Call,
                                         CheckerContext &C) const {
  const FunctionDecl *FuncDecl = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FuncDecl)
    return;

  ProgramStateRef State = C.getState();

  std::vector<std::function<std::string(BugReport & BR)>> Notes;
  SymbolRef ResultSymbol = nullptr;
  if (const auto *TypeDefTy = FuncDecl->getReturnType()->getAs<TypedefType>())
    if (TypeDefTy->getDecl()->getName() == ErrorTypeName)
      ResultSymbol = Call.getReturnValue().getAsSymbol();

  // Function returns an open handle.
  if (hasFuchsiaAttr<AcquireHandleAttr>(FuncDecl)) {
    SymbolRef RetSym = Call.getReturnValue().getAsSymbol();
    Notes.push_back([RetSym, FuncDecl](BugReport &BR) -> std::string {
      auto *PathBR = static_cast<PathSensitiveBugReport *>(&BR);
      if (auto IsInteresting = PathBR->getInterestingnessKind(RetSym)) {
        std::string SBuf;
        llvm::raw_string_ostream OS(SBuf);
        OS << "Function '" << FuncDecl->getNameAsString()
           << "' returns an open handle";
        return OS.str();
      } else
        return "";
    });
    State =
        State->set<HStateMap>(RetSym, HandleState::getMaybeAllocated(nullptr));
  }

  for (unsigned Arg = 0; Arg < Call.getNumArgs(); ++Arg) {
    if (Arg >= FuncDecl->getNumParams())
      break;
    const ParmVarDecl *PVD = FuncDecl->getParamDecl(Arg);
    unsigned ParamDiagIdx = PVD->getFunctionScopeIndex() + 1;
    SymbolRef Handle =
        getFuchsiaHandleSymbol(PVD->getType(), Call.getArgSVal(Arg), State);
    if (!Handle)
      continue;

    const HandleState *HState = State->get<HStateMap>(Handle);
    if (HState && HState->isEscaped())
      continue;
    if (hasFuchsiaAttr<ReleaseHandleAttr>(PVD)) {
      if (HState && HState->isReleased()) {
        reportDoubleRelease(Handle, Call.getArgSourceRange(Arg), C);
        return;
      } else {
        Notes.push_back([Handle, ParamDiagIdx](BugReport &BR) -> std::string {
          auto *PathBR = static_cast<PathSensitiveBugReport *>(&BR);
          if (auto IsInteresting = PathBR->getInterestingnessKind(Handle)) {
            std::string SBuf;
            llvm::raw_string_ostream OS(SBuf);
            OS << "Handle released through " << ParamDiagIdx
               << llvm::getOrdinalSuffix(ParamDiagIdx) << " parameter";
            return OS.str();
          } else
            return "";
        });
        State = State->set<HStateMap>(Handle, HandleState::getReleased());
      }
    } else if (hasFuchsiaAttr<AcquireHandleAttr>(PVD)) {
      Notes.push_back([Handle, ParamDiagIdx](BugReport &BR) -> std::string {
        auto *PathBR = static_cast<PathSensitiveBugReport *>(&BR);
        if (auto IsInteresting = PathBR->getInterestingnessKind(Handle)) {
          std::string SBuf;
          llvm::raw_string_ostream OS(SBuf);
          OS << "Handle allocated through " << ParamDiagIdx
             << llvm::getOrdinalSuffix(ParamDiagIdx) << " parameter";
          return OS.str();
        } else
          return "";
      });
      State = State->set<HStateMap>(
          Handle, HandleState::getMaybeAllocated(ResultSymbol));
    }
  }
  const NoteTag *T = nullptr;
  if (!Notes.empty()) {
    T = C.getNoteTag([this, Notes{std::move(Notes)}](
                         PathSensitiveBugReport &BR) -> std::string {
          if (&BR.getBugType() != &UseAfterReleaseBugType &&
              &BR.getBugType() != &LeakBugType &&
              &BR.getBugType() != &DoubleReleaseBugType)
            return "";
          for (auto &Note : Notes) {
            std::string Text = Note(BR);
            if (!Text.empty())
              return Text;
          }
          return "";
        });
  }
  C.addTransition(State, T);
}

void FuchsiaHandleChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                            CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SmallVector<SymbolRef, 2> LeakedSyms;
  HStateMapTy TrackedHandles = State->get<HStateMap>();
  for (auto &CurItem : TrackedHandles) {
    SymbolRef ErrorSym = CurItem.second.getErrorSym();
    // Keeping zombie handle symbols. In case the error symbol is dying later
    // than the handle symbol we might produce spurious leak warnings (in case
    // we find out later from the status code that the handle allocation failed
    // in the first place).
    if (!SymReaper.isDead(CurItem.first) ||
        (ErrorSym && !SymReaper.isDead(ErrorSym)))
      continue;
    if (CurItem.second.isAllocated() || CurItem.second.maybeAllocated())
      LeakedSyms.push_back(CurItem.first);
    State = State->remove<HStateMap>(CurItem.first);
  }

  ExplodedNode *N = C.getPredecessor();
  if (!LeakedSyms.empty())
    N = reportLeaks(LeakedSyms, C, N);

  C.addTransition(State, N);
}

// Acquiring a handle is not always successful. In Fuchsia most functions
// return a status code that determines the status of the handle.
// When we split the path based on this status code we know that on one
// path we do have the handle and on the other path the acquire failed.
// This method helps avoiding false positive leak warnings on paths where
// the function failed.
// Moreover, when a handle is known to be zero (the invalid handle),
// we no longer can follow the symbol on the path, becaue the constant
// zero will be used instead of the symbol. We also do not need to release
// an invalid handle, so we remove the corresponding symbol from the state.
ProgramStateRef FuchsiaHandleChecker::evalAssume(ProgramStateRef State,
                                                 SVal Cond,
                                                 bool Assumption) const {
  // TODO: add notes about successes/fails for APIs.
  ConstraintManager &Cmr = State->getConstraintManager();
  HStateMapTy TrackedHandles = State->get<HStateMap>();
  for (auto &CurItem : TrackedHandles) {
    ConditionTruthVal HandleVal = Cmr.isNull(State, CurItem.first);
    if (HandleVal.isConstrainedTrue()) {
      // The handle is invalid. We can no longer follow the symbol on this path.
      State = State->remove<HStateMap>(CurItem.first);
    }
    SymbolRef ErrorSym = CurItem.second.getErrorSym();
    if (!ErrorSym)
      continue;
    ConditionTruthVal ErrorVal = Cmr.isNull(State, ErrorSym);
    if (ErrorVal.isConstrainedTrue()) {
      // Allocation succeeded.
      if (CurItem.second.maybeAllocated())
        State = State->set<HStateMap>(
            CurItem.first, HandleState::getAllocated(State, CurItem.second));
    } else if (ErrorVal.isConstrainedFalse()) {
      // Allocation failed.
      if (CurItem.second.maybeAllocated())
        State = State->remove<HStateMap>(CurItem.first);
    }
  }
  return State;
}

ProgramStateRef FuchsiaHandleChecker::checkPointerEscape(
    ProgramStateRef State, const InvalidatedSymbols &Escaped,
    const CallEvent *Call, PointerEscapeKind Kind) const {
  const FunctionDecl *FuncDecl =
      Call ? dyn_cast_or_null<FunctionDecl>(Call->getDecl()) : nullptr;

  llvm::DenseSet<SymbolRef> UnEscaped;
  // Not all calls should escape our symbols.
  if (FuncDecl &&
      (Kind == PSK_DirectEscapeOnCall || Kind == PSK_IndirectEscapeOnCall ||
       Kind == PSK_EscapeOutParameters)) {
    for (unsigned Arg = 0; Arg < Call->getNumArgs(); ++Arg) {
      if (Arg >= FuncDecl->getNumParams())
        break;
      const ParmVarDecl *PVD = FuncDecl->getParamDecl(Arg);
      SymbolRef Handle =
          getFuchsiaHandleSymbol(PVD->getType(), Call->getArgSVal(Arg), State);
      if (!Handle)
        continue;
      if (hasFuchsiaAttr<UseHandleAttr>(PVD) ||
          hasFuchsiaAttr<ReleaseHandleAttr>(PVD))
        UnEscaped.insert(Handle);
    }
  }

  // For out params, we have to deal with derived symbols. See
  // MacOSKeychainAPIChecker for details.
  for (auto I : State->get<HStateMap>()) {
    if (Escaped.count(I.first) && !UnEscaped.count(I.first))
      State = State->set<HStateMap>(I.first, HandleState::getEscaped());
    if (const auto *SD = dyn_cast<SymbolDerived>(I.first)) {
      auto ParentSym = SD->getParentSymbol();
      if (Escaped.count(ParentSym))
        State = State->set<HStateMap>(I.first, HandleState::getEscaped());
    }
  }

  return State;
}

ExplodedNode *
FuchsiaHandleChecker::reportLeaks(ArrayRef<SymbolRef> LeakedHandles,
                                  CheckerContext &C, ExplodedNode *Pred) const {
  ExplodedNode *ErrNode = C.generateNonFatalErrorNode(C.getState(), Pred);
  for (SymbolRef LeakedHandle : LeakedHandles) {
    reportBug(LeakedHandle, ErrNode, C, nullptr, LeakBugType,
              "Potential leak of handle");
  }
  return ErrNode;
}

void FuchsiaHandleChecker::reportDoubleRelease(SymbolRef HandleSym,
                                               const SourceRange &Range,
                                               CheckerContext &C) const {
  ExplodedNode *ErrNode = C.generateErrorNode(C.getState());
  reportBug(HandleSym, ErrNode, C, &Range, DoubleReleaseBugType,
            "Releasing a previously released handle");
}

void FuchsiaHandleChecker::reportUseAfterFree(SymbolRef HandleSym,
                                              const SourceRange &Range,
                                              CheckerContext &C) const {
  ExplodedNode *ErrNode = C.generateErrorNode(C.getState());
  reportBug(HandleSym, ErrNode, C, &Range, UseAfterReleaseBugType,
            "Using a previously released handle");
}

void FuchsiaHandleChecker::reportBug(SymbolRef Sym, ExplodedNode *ErrorNode,
                                     CheckerContext &C,
                                     const SourceRange *Range,
                                     const BugType &Type, StringRef Msg) const {
  if (!ErrorNode)
    return;

  std::unique_ptr<PathSensitiveBugReport> R;
  if (Type.isSuppressOnSink()) {
    const ExplodedNode *AcquireNode = getAcquireSite(ErrorNode, Sym, C);
    if (AcquireNode) {
      PathDiagnosticLocation LocUsedForUniqueing =
          PathDiagnosticLocation::createBegin(
              AcquireNode->getStmtForDiagnostics(), C.getSourceManager(),
              AcquireNode->getLocationContext());

      R = std::make_unique<PathSensitiveBugReport>(
          Type, Msg, ErrorNode, LocUsedForUniqueing,
          AcquireNode->getLocationContext()->getDecl());
    }
  }
  if (!R)
    R = std::make_unique<PathSensitiveBugReport>(Type, Msg, ErrorNode);
  if (Range)
    R->addRange(*Range);
  R->markInteresting(Sym);
  C.emitReport(std::move(R));
}

void ento::registerFuchsiaHandleChecker(CheckerManager &mgr) {
  mgr.registerChecker<FuchsiaHandleChecker>();
}

bool ento::shouldRegisterFuchsiaHandleChecker(const CheckerManager &mgr) {
  return true;
}

void FuchsiaHandleChecker::printState(raw_ostream &Out, ProgramStateRef State,
                                      const char *NL, const char *Sep) const {

  HStateMapTy StateMap = State->get<HStateMap>();

  if (!StateMap.isEmpty()) {
    Out << Sep << "FuchsiaHandleChecker :" << NL;
    for (HStateMapTy::iterator I = StateMap.begin(), E = StateMap.end(); I != E;
         ++I) {
      I.getKey()->dumpToStream(Out);
      Out << " : ";
      I.getData().dump(Out);
      Out << NL;
    }
  }
}
