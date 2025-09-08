//== ValistChecker.cpp - stdarg.h macro usage checker -----------*- C++ -*--==//
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

using namespace clang;
using namespace ento;

REGISTER_SET_WITH_PROGRAMSTATE(InitializedVALists, const MemRegion *)

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
                                     bool &IsSymbolic, CheckerContext &C) const;
  const ExplodedNode *getStartCallSite(const ExplodedNode *N,
                                       const MemRegion *Reg) const;

  void reportUninitializedAccess(const MemRegion *VAList, StringRef Msg,
                                 CheckerContext &C) const;
  void reportLeaked(const RegionVector &Leaked, StringRef Msg1, StringRef Msg2,
                    CheckerContext &C, ExplodedNode *N) const;

  void checkVAListStartCall(const CallEvent &Call, CheckerContext &C,
                            bool IsCopy) const;
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
    checkVAListStartCall(Call, C, false);
  else if (VaCopy.matches(Call))
    checkVAListStartCall(Call, C, true);
  else if (VaEnd.matches(Call))
    checkVAListEndCall(Call, C);
  else {
    for (auto FuncInfo : VAListAccepters) {
      if (!FuncInfo.Func.matches(Call))
        continue;
      bool Symbolic;
      const MemRegion *VAList =
          getVAListAsRegion(Call.getArgSVal(FuncInfo.ParamIndex),
                            Call.getArgExpr(FuncInfo.ParamIndex), Symbolic, C);
      if (!VAList)
        return;

      if (C.getState()->contains<InitializedVALists>(VAList))
        return;

      // We did not see va_start call, but the source of the region is unknown.
      // Be conservative and assume the best.
      if (Symbolic)
        return;

      SmallString<80> Errmsg("Function '");
      Errmsg += FuncInfo.Func.getFunctionName();
      Errmsg += "' is called with an uninitialized va_list argument";
      reportUninitializedAccess(VAList, Errmsg.c_str(), C);
      break;
    }
  }
}

const MemRegion *VAListChecker::getVAListAsRegion(SVal SV, const Expr *E,
                                                  bool &IsSymbolic,
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
  IsSymbolic = Reg && Reg->getBaseRegion()->getAs<SymbolicRegion>();
  // Some VarRegion based VA lists reach here as ElementRegions.
  const auto *EReg = dyn_cast_or_null<ElementRegion>(Reg);
  return (EReg && VAListModelledAsArray) ? EReg->getSuperRegion() : Reg;
}

void VAListChecker::checkPreStmt(const VAArgExpr *VAA,
                                 CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const Expr *ArgExpr = VAA->getSubExpr();
  SVal VAListSVal = C.getSVal(ArgExpr);
  bool Symbolic;
  const MemRegion *VAList = getVAListAsRegion(VAListSVal, ArgExpr, Symbolic, C);
  if (!VAList)
    return;
  if (Symbolic)
    return;
  if (!State->contains<InitializedVALists>(VAList))
    reportUninitializedAccess(
        VAList, "va_arg() is called on an uninitialized va_list", C);
}

void VAListChecker::checkDeadSymbols(SymbolReaper &SR,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  InitializedVAListsTy Tracked = State->get<InitializedVALists>();
  RegionVector Leaked;
  for (const MemRegion *Reg : Tracked) {
    if (SR.isLiveRegion(Reg))
      continue;
    Leaked.push_back(Reg);
    State = State->remove<InitializedVALists>(Reg);
  }
  if (ExplodedNode *N = C.addTransition(State))
    reportLeaked(Leaked, "Initialized va_list", " is leaked", C, N);
}

// This function traverses the exploded graph backwards and finds the node where
// the va_list is initialized. That node is used for uniquing the bug paths.
// It is not likely that there are several different va_lists that belongs to
// different stack frames, so that case is not yet handled.
const ExplodedNode *
VAListChecker::getStartCallSite(const ExplodedNode *N,
                                const MemRegion *Reg) const {
  const LocationContext *LeakContext = N->getLocationContext();
  const ExplodedNode *StartCallNode = N;

  bool FoundInitializedState = false;

  while (N) {
    ProgramStateRef State = N->getState();
    if (!State->contains<InitializedVALists>(Reg)) {
      if (FoundInitializedState)
        break;
    } else {
      FoundInitializedState = true;
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
                                         CheckerContext &C, bool IsCopy) const {
  bool Symbolic;
  const MemRegion *VAList =
      getVAListAsRegion(Call.getArgSVal(0), Call.getArgExpr(0), Symbolic, C);
  if (!VAList)
    return;

  ProgramStateRef State = C.getState();

  if (IsCopy) {
    const MemRegion *Arg2 =
        getVAListAsRegion(Call.getArgSVal(1), Call.getArgExpr(1), Symbolic, C);
    if (Arg2) {
      if (VAList == Arg2) {
        RegionVector Leaked{VAList};
        if (ExplodedNode *N = C.addTransition(State))
          reportLeaked(Leaked, "va_list", " is copied onto itself", C, N);
        return;
      }
      if (!State->contains<InitializedVALists>(Arg2) && !Symbolic) {
        if (State->contains<InitializedVALists>(VAList)) {
          State = State->remove<InitializedVALists>(VAList);
          RegionVector Leaked{VAList};
          if (ExplodedNode *N = C.addTransition(State))
            reportLeaked(Leaked, "Initialized va_list",
                         " is overwritten by an uninitialized one", C, N);
        } else {
          reportUninitializedAccess(Arg2, "Uninitialized va_list is copied", C);
        }
        return;
      }
    }
  }
  if (State->contains<InitializedVALists>(VAList)) {
    RegionVector Leaked{VAList};
    if (ExplodedNode *N = C.addTransition(State))
      reportLeaked(Leaked, "Initialized va_list", " is initialized again", C,
                   N);
    return;
  }

  State = State->add<InitializedVALists>(VAList);
  C.addTransition(State);
}

void VAListChecker::checkVAListEndCall(const CallEvent &Call,
                                       CheckerContext &C) const {
  bool Symbolic;
  const MemRegion *VAList =
      getVAListAsRegion(Call.getArgSVal(0), Call.getArgExpr(0), Symbolic, C);
  if (!VAList)
    return;

  // We did not see va_start call, but the source of the region is unknown.
  // Be conservative and assume the best.
  if (Symbolic)
    return;

  if (!C.getState()->contains<InitializedVALists>(VAList)) {
    reportUninitializedAccess(
        VAList, "va_end() is called on an uninitialized va_list", C);
    return;
  }
  ProgramStateRef State = C.getState();
  State = State->remove<InitializedVALists>(VAList);
  C.addTransition(State);
}

PathDiagnosticPieceRef VAListChecker::VAListBugVisitor::VisitNode(
    const ExplodedNode *N, BugReporterContext &BRC, PathSensitiveBugReport &) {
  ProgramStateRef State = N->getState();
  ProgramStateRef StatePrev = N->getFirstPred()->getState();

  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  StringRef Msg;
  if (State->contains<InitializedVALists>(Reg) &&
      !StatePrev->contains<InitializedVALists>(Reg))
    Msg = "Initialized va_list";
  else if (!State->contains<InitializedVALists>(Reg) &&
           StatePrev->contains<InitializedVALists>(Reg))
    Msg = "Ended va_list";

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
