//== VAListChecker.cpp - stdarg.h macro usage checker -----------*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines checkers which detect usage of uninitialized va_list values
// and va_start calls with no matching va_end.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang;
using namespace ento;
using llvm::formatv;

namespace {
enum class VAListState {
  Uninitialized,
  Unknown,
  Initialized,
  Released,
};

constexpr llvm::StringLiteral StateNames[] = {
    "uninitialized", "unknown", "initialized", "already released"};
} // end anonymous namespace

static StringRef describeState(const VAListState S) {
  return StateNames[static_cast<int>(S)];
}

REGISTER_MAP_WITH_PROGRAMSTATE(VAListStateMap, const MemRegion *, VAListState)

static VAListState getVAListState(ProgramStateRef State, const MemRegion *Reg) {
  if (const VAListState *Res = State->get<VAListStateMap>(Reg))
    return *Res;
  return Reg->getSymbolicBase() ? VAListState::Unknown
                                : VAListState::Uninitialized;
}

namespace {
typedef SmallVector<const MemRegion *, 2> RegionVector;

class VAListChecker : public Checker<check::PreCall, check::PreStmt<VAArgExpr>,
                                     check::DeadSymbols> {
  const BugType LeakBug{this, "Leaked va_list", categories::MemoryError,
                        /*SuppressOnSink=*/true};
  const BugType UninitAccessBug{this, "Uninitialized va_list",
                                categories::MemoryError};

  struct VAListAccepter {
    CallDescription Func;
    int ParamIndex;
  };
  static const SmallVector<VAListAccepter, 15> VAListAccepters;
  static const CallDescription VaStart, VaEnd, VaCopy;

public:
  void checkPreStmt(const VAArgExpr *VAA, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;

private:
  const MemRegion *getVAListAsRegion(SVal SV, const Expr *VAExpr,
                                     CheckerContext &C) const;
  const ExplodedNode *getStartCallSite(const ExplodedNode *N,
                                       const MemRegion *Reg) const;

  void reportUninitializedAccess(const MemRegion *VAList, StringRef Msg,
                                 CheckerContext &C) const;
  void reportLeaked(const RegionVector &Leaked, StringRef Msg1, StringRef Msg2,
                    CheckerContext &C, ExplodedNode *N) const;

  void checkVAListStartCall(const CallEvent &Call, CheckerContext &C) const;
  void checkVAListCopyCall(const CallEvent &Call, CheckerContext &C) const;
  void checkVAListEndCall(const CallEvent &Call, CheckerContext &C) const;

  class VAListBugVisitor : public BugReporterVisitor {
  public:
    VAListBugVisitor(const MemRegion *Reg, bool IsLeak = false)
        : Reg(Reg), IsLeak(IsLeak) {}
    void Profile(llvm::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
      ID.AddPointer(Reg);
    }
    PathDiagnosticPieceRef getEndPath(BugReporterContext &BRC,
                                      const ExplodedNode *EndPathNode,
                                      PathSensitiveBugReport &BR) override {
      if (!IsLeak)
        return nullptr;

      PathDiagnosticLocation L = BR.getLocation();
      // Do not add the statement itself as a range in case of leak.
      return std::make_shared<PathDiagnosticEventPiece>(L, BR.getDescription(),
                                                        false);
    }
    PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                     BugReporterContext &BRC,
                                     PathSensitiveBugReport &BR) override;

  private:
    const MemRegion *Reg;
    bool IsLeak;
  };
};

const SmallVector<VAListChecker::VAListAccepter, 15>
    VAListChecker::VAListAccepters = {{{CDM::CLibrary, {"vfprintf"}, 3}, 2},
                                      {{CDM::CLibrary, {"vfscanf"}, 3}, 2},
                                      {{CDM::CLibrary, {"vprintf"}, 2}, 1},
                                      {{CDM::CLibrary, {"vscanf"}, 2}, 1},
                                      {{CDM::CLibrary, {"vsnprintf"}, 4}, 3},
                                      {{CDM::CLibrary, {"vsprintf"}, 3}, 2},
                                      {{CDM::CLibrary, {"vsscanf"}, 3}, 2},
                                      {{CDM::CLibrary, {"vfwprintf"}, 3}, 2},
                                      {{CDM::CLibrary, {"vfwscanf"}, 3}, 2},
                                      {{CDM::CLibrary, {"vwprintf"}, 2}, 1},
                                      {{CDM::CLibrary, {"vwscanf"}, 2}, 1},
                                      {{CDM::CLibrary, {"vswprintf"}, 4}, 3},
                                      // vswprintf is the wide version of
                                      // vsnprintf, vsprintf has no wide version
                                      {{CDM::CLibrary, {"vswscanf"}, 3}, 2}};

const CallDescription VAListChecker::VaStart(CDM::CLibrary,
                                             {"__builtin_va_start"}, /*Args=*/2,
                                             /*Params=*/1),
    VAListChecker::VaCopy(CDM::CLibrary, {"__builtin_va_copy"}, 2),
    VAListChecker::VaEnd(CDM::CLibrary, {"__builtin_va_end"}, 1);
} // end anonymous namespace

void VAListChecker::checkPreCall(const CallEvent &Call,
                                 CheckerContext &C) const {
  if (VaStart.matches(Call))
    checkVAListStartCall(Call, C);
  else if (VaCopy.matches(Call))
    checkVAListCopyCall(Call, C);
  else if (VaEnd.matches(Call))
    checkVAListEndCall(Call, C);
  else {
    for (auto FuncInfo : VAListAccepters) {
      if (!FuncInfo.Func.matches(Call))
        continue;
      const MemRegion *VAList =
          getVAListAsRegion(Call.getArgSVal(FuncInfo.ParamIndex),
                            Call.getArgExpr(FuncInfo.ParamIndex), C);
      if (!VAList)
        return;
      VAListState S = getVAListState(C.getState(), VAList);

      if (S == VAListState::Initialized || S == VAListState::Unknown)
        return;

      std::string ErrMsg =
          formatv("Function '{0}' is called with an {1} va_list argument",
                  FuncInfo.Func.getFunctionName(), describeState(S));
      reportUninitializedAccess(VAList, ErrMsg, C);
      break;
    }
  }
}

const MemRegion *VAListChecker::getVAListAsRegion(SVal SV, const Expr *E,
                                                  CheckerContext &C) const {
  const MemRegion *Reg = SV.getAsRegion();
  if (!Reg)
    return nullptr;
  // TODO: In the future this should be abstracted away by the analyzer.
  bool VAListModelledAsArray = false;
  if (const auto *Cast = dyn_cast<CastExpr>(E)) {
    QualType Ty = Cast->getType();
    VAListModelledAsArray =
        Ty->isPointerType() && Ty->getPointeeType()->isRecordType();
  }
  if (const auto *DeclReg = Reg->getAs<DeclRegion>()) {
    if (isa<ParmVarDecl>(DeclReg->getDecl()))
      Reg = C.getState()->getSVal(SV.castAs<Loc>()).getAsRegion();
  }
  // Some VarRegion based VA lists reach here as ElementRegions.
  const auto *EReg = dyn_cast_or_null<ElementRegion>(Reg);
  return (EReg && VAListModelledAsArray) ? EReg->getSuperRegion() : Reg;
}

void VAListChecker::checkPreStmt(const VAArgExpr *VAA,
                                 CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const Expr *ArgExpr = VAA->getSubExpr();
  const MemRegion *VAList = getVAListAsRegion(C.getSVal(ArgExpr), ArgExpr, C);
  if (!VAList)
    return;
  VAListState S = getVAListState(C.getState(), VAList);
  if (S == VAListState::Initialized || S == VAListState::Unknown)
    return;

  std::string ErrMsg =
      formatv("va_arg() is called on an {0} va_list", describeState(S));
  reportUninitializedAccess(VAList, ErrMsg, C);
}

void VAListChecker::checkDeadSymbols(SymbolReaper &SR,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  VAListStateMapTy Tracked = State->get<VAListStateMap>();
  RegionVector Leaked;
  for (const auto &[Reg, S] : Tracked) {
    if (SR.isLiveRegion(Reg))
      continue;
    if (S == VAListState::Initialized)
      Leaked.push_back(Reg);
    State = State->remove<VAListStateMap>(Reg);
  }
  if (ExplodedNode *N = C.addTransition(State)) {
    reportLeaked(Leaked, "Initialized va_list", " is leaked", C, N);
  }
}

// This function traverses the exploded graph backwards and finds the node where
// the va_list becomes initialized. That node is used for uniquing the bug
// paths. It is not likely that there are several different va_lists that
// belongs to different stack frames, so that case is not yet handled.
const ExplodedNode *
VAListChecker::getStartCallSite(const ExplodedNode *N,
                                const MemRegion *Reg) const {
  const LocationContext *LeakContext = N->getLocationContext();
  const ExplodedNode *StartCallNode = N;

  bool SeenInitializedState = false;

  while (N) {
    VAListState S = getVAListState(N->getState(), Reg);
    if (S == VAListState::Initialized) {
      SeenInitializedState = true;
    } else if (SeenInitializedState) {
      break;
    }
    const LocationContext *NContext = N->getLocationContext();
    if (NContext == LeakContext || NContext->isParentOf(LeakContext))
      StartCallNode = N;
    N = N->pred_empty() ? nullptr : *(N->pred_begin());
  }

  return StartCallNode;
}

void VAListChecker::reportUninitializedAccess(const MemRegion *VAList,
                                              StringRef Msg,
                                              CheckerContext &C) const {
  if (ExplodedNode *N = C.generateErrorNode()) {
    auto R = std::make_unique<PathSensitiveBugReport>(UninitAccessBug, Msg, N);
    R->markInteresting(VAList);
    R->addVisitor(std::make_unique<VAListBugVisitor>(VAList));
    C.emitReport(std::move(R));
  }
}

void VAListChecker::reportLeaked(const RegionVector &Leaked, StringRef Msg1,
                                 StringRef Msg2, CheckerContext &C,
                                 ExplodedNode *N) const {
  for (const MemRegion *Reg : Leaked) {
    const ExplodedNode *StartNode = getStartCallSite(N, Reg);
    PathDiagnosticLocation LocUsedForUniqueing;

    if (const Stmt *StartCallStmt = StartNode->getStmtForDiagnostics())
      LocUsedForUniqueing = PathDiagnosticLocation::createBegin(
          StartCallStmt, C.getSourceManager(), StartNode->getLocationContext());

    SmallString<100> Buf;
    llvm::raw_svector_ostream OS(Buf);
    OS << Msg1;
    std::string VariableName = Reg->getDescriptiveName();
    if (!VariableName.empty())
      OS << " " << VariableName;
    OS << Msg2;

    auto R = std::make_unique<PathSensitiveBugReport>(
        LeakBug, OS.str(), N, LocUsedForUniqueing,
        StartNode->getLocationContext()->getDecl());
    R->markInteresting(Reg);
    R->addVisitor(std::make_unique<VAListBugVisitor>(Reg, true));
    C.emitReport(std::move(R));
  }
}

void VAListChecker::checkVAListStartCall(const CallEvent &Call,
                                         CheckerContext &C) const {
  const MemRegion *Arg =
      getVAListAsRegion(Call.getArgSVal(0), Call.getArgExpr(0), C);
  if (!Arg)
    return;

  ProgramStateRef State = C.getState();
  VAListState ArgState = getVAListState(State, Arg);

  if (ArgState == VAListState::Initialized) {
    RegionVector Leaked{Arg};
    if (ExplodedNode *N = C.addTransition(State))
      reportLeaked(Leaked, "Initialized va_list", " is initialized again", C,
                   N);
    return;
  }

  State = State->set<VAListStateMap>(Arg, VAListState::Initialized);
  C.addTransition(State);
}

void VAListChecker::checkVAListCopyCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  const MemRegion *Arg1 =
      getVAListAsRegion(Call.getArgSVal(0), Call.getArgExpr(0), C);
  const MemRegion *Arg2 =
      getVAListAsRegion(Call.getArgSVal(1), Call.getArgExpr(1), C);
  if (!Arg1 || !Arg2)
    return;

  ProgramStateRef State = C.getState();
  if (Arg1 == Arg2) {
    RegionVector Leaked{Arg1};
    if (ExplodedNode *N = C.addTransition(State))
      reportLeaked(Leaked, "va_list", " is copied onto itself", C, N);
    return;
  }
  VAListState State1 = getVAListState(State, Arg1);
  VAListState State2 = getVAListState(State, Arg2);
  // Update the ProgramState by copying the state of Arg2 to Arg1.
  State = State->set<VAListStateMap>(Arg1, State2);
  if (State1 == VAListState::Initialized) {
    RegionVector Leaked{Arg1};
    std::string Msg2 =
        formatv(" is overwritten by {0} {1} one",
                (State2 == VAListState::Initialized) ? "another" : "an",
                describeState(State2));
    if (ExplodedNode *N = C.addTransition(State))
      reportLeaked(Leaked, "Initialized va_list", Msg2, C, N);
    return;
  }
  if (State2 != VAListState::Initialized && State2 != VAListState::Unknown) {
    std::string Msg = formatv("{0} va_list is copied", describeState(State2));
    Msg[0] = toupper(Msg[0]);
    reportUninitializedAccess(Arg2, Msg, C);
    return;
  }
  C.addTransition(State);
}

void VAListChecker::checkVAListEndCall(const CallEvent &Call,
                                       CheckerContext &C) const {
  const MemRegion *Arg =
      getVAListAsRegion(Call.getArgSVal(0), Call.getArgExpr(0), C);
  if (!Arg)
    return;

  ProgramStateRef State = C.getState();
  VAListState ArgState = getVAListState(State, Arg);

  if (ArgState != VAListState::Unknown &&
      ArgState != VAListState::Initialized) {
    std::string Msg = formatv("va_end() is called on an {0} va_list",
                              describeState(ArgState));
    reportUninitializedAccess(Arg, Msg, C);
    return;
  }
  State = State->set<VAListStateMap>(Arg, VAListState::Released);
  C.addTransition(State);
}

PathDiagnosticPieceRef VAListChecker::VAListBugVisitor::VisitNode(
    const ExplodedNode *N, BugReporterContext &BRC, PathSensitiveBugReport &) {
  ProgramStateRef State = N->getState();
  ProgramStateRef StatePrev = N->getFirstPred()->getState();

  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  VAListState After = getVAListState(State, Reg);
  VAListState Before = getVAListState(StatePrev, Reg);
  if (Before == After)
    return nullptr;

  StringRef Msg;
  switch (After) {
  case VAListState::Uninitialized:
    Msg = "Copied uninitialized contents into the va_list";
    break;
  case VAListState::Unknown:
    Msg = "Copied unknown contents into the va_list";
    break;
  case VAListState::Initialized:
    Msg = "Initialized va_list";
    break;
  case VAListState::Released:
    Msg = "Ended va_list";
    break;
  }

  if (Msg.empty())
    return nullptr;

  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                             N->getLocationContext());
  return std::make_shared<PathDiagnosticEventPiece>(Pos, Msg, true);
}

void ento::registerVAListChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<VAListChecker>();
}

bool ento::shouldRegisterVAListChecker(const CheckerManager &) { return true; }
