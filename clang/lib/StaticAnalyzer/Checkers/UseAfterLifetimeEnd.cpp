#include "clang/AST/Attr.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(LifetimeSourceSet, const MemRegion *)
REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMap, SVal, LifetimeSourceSet)

namespace {
class UseAfterLifetimeEnd
    : public Checker<check::PostCall, check::EndFunction, check::DeadSymbols> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;
  void reportDanglingSource(const MemRegion *Region, ExplodedNode *N,
                            CheckerContext &C) const;
  void checkReturnedBorrower(SVal Val, ProgramStateRef State,
                             CheckerContext &C) const;
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  const BugType BugMsg{this, "UseAfterLifetimeEnd", "LifetimeBound"};
};

} // namespace

static ProgramStateRef bindValues(ProgramStateRef State, SVal RetVal,
                                  const MemRegion *Source) {
  LifetimeSourceSet::Factory &F = State->get_context<LifetimeSourceSet>();

  const LifetimeSourceSet *LSet = State->get<LifetimeBoundMap>(RetVal);
  LifetimeSourceSet Set = LSet ? *LSet : F.getEmptySet();
  Set = F.add(Set, Source);
  State = State->set<LifetimeBoundMap>(RetVal, Set);
  return State;
}

void UseAfterLifetimeEnd::checkPostCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  const auto *FC = dyn_cast<AnyFunctionCall>(&Call);
  if (!FC)
    return;

  const FunctionDecl *FD = FC->getDecl();
  if (!FD)
    return;

  SVal RetVal = Call.getReturnValue();

  for (const ParmVarDecl *PVD : FD->parameters()) {
    if (PVD->hasAttr<LifetimeBoundAttr>()) {
      unsigned Idx = PVD->getFunctionScopeIndex();
      SVal Arg = Call.getArgSVal(Idx);
      if (const MemRegion *ArgValRegion = Arg.getAsRegion())
        State = bindValues(State, RetVal, ArgValRegion);
    }
  }

  if (const auto *IC = dyn_cast<CXXInstanceCall>(&Call)) {
    if (lifetimes::implicitObjectParamIsLifetimeBound(FD)) {
      if (const MemRegion *AttrRegion = IC->getCXXThisVal().getAsRegion()) {
        State = bindValues(State, RetVal, AttrRegion);
      }
    }
  }
  C.addTransition(State);
}

static bool hasDanglingSource(const MemRegion *Source, ProgramStateRef State,
                              CheckerContext &C) {
  // FIXME: The checker currently handles stack-region sources. Other
  // region kinds require separate methodology. For example, heap
  // regions do not go out of scope at the end of a stack frame, so
  // in order to detect those type of dangling sources the function
  // needs to be expanded to an event-driven approach as well.
  if (const auto *StackSpace =
          Source->getMemorySpaceAs<StackSpaceRegion>(State)) {
    const StackFrame *SF = StackSpace->getStackFrame();
    const StackFrame *CurrentSF = C.getStackFrame();
    if (SF == CurrentSF || !SF->isParentOf(CurrentSF))
      return true;
  }
  return false;
}

void UseAfterLifetimeEnd::checkReturnedBorrower(SVal Val, ProgramStateRef State,
                                                CheckerContext &C) const {
  if (auto *SourceSet = State->get<LifetimeBoundMap>(Val)) {
    ExplodedNode *N = nullptr;
    for (const MemRegion *Region : *SourceSet) {
      if (hasDanglingSource(Region, State, C)) {
        if (!N)
          N = C.generateNonFatalErrorNode();
        if (!N)
          return;
        reportDanglingSource(Region, N, C);
      }
    }
  }
}

void UseAfterLifetimeEnd::checkEndFunction(const ReturnStmt *RS,
                                           CheckerContext &C) const {
  if (!RS)
    return;

  ProgramStateRef State = C.getState();
  auto LBMap = State->get<LifetimeBoundMap>();

  if (LBMap.isEmpty())
    return;

  const Expr *RetExpr = RS->getRetValue();
  if (!RetExpr)
    return;

  RetExpr = RetExpr->IgnoreParens();
  SVal RetVal = C.getSVal(RetExpr);
  checkReturnedBorrower(RetVal, State, C);
}

void UseAfterLifetimeEnd::reportDanglingSource(const MemRegion *Region,
                                               ExplodedNode *N,
                                               CheckerContext &C) const {
  auto BR = std::make_unique<PathSensitiveBugReport>(
      BugMsg,
      (llvm::Twine("Returning value bound to '") + Region->getString() +
       "' that will go out of scope"),
      N);
  C.emitReport(std::move(BR));
}

void UseAfterLifetimeEnd::checkDeadSymbols(SymbolReaper &SymReaper,
                                           CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  LifetimeBoundMapTy LBMap = State->get<LifetimeBoundMap>();

  for (SVal Val : llvm::make_first_range(LBMap)) {
    if (const MemRegion *ValRegion = Val.getAsRegion()) {
      if (!SymReaper.isLiveRegion(ValRegion))
        State = State->remove<LifetimeBoundMap>(Val);
    } else if (SymbolRef ValRef =
                   Val.getAsSymbol(/*IncludeBaseRegions=*/true)) {
      if (!SymReaper.isLive(ValRef))
        State = State->remove<LifetimeBoundMap>(Val);
    }
  }

  C.addTransition(State);
}

void UseAfterLifetimeEnd::printState(raw_ostream &Out, ProgramStateRef State,
                                     const char *NL, const char *Sep) const {
  auto LBMap = State->get<LifetimeBoundMap>();

  if (LBMap.isEmpty())
    return;

  Out << Sep << "LifetimeBound bindings:" << NL;
  for (auto &&[OriginSym, SourceSet] : LBMap) {
    for (const auto *Region : SourceSet)
      Out << " Origin " << OriginSym << " contains Loan " << Region << NL;
  }
}

namespace {
class DebugUseAfterLifetimeEnd : public Checker<eval::Call> {
public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void analyzerDumpLifetimeOriginsOf(const CallEvent &Call,
                                     CheckerContext &C) const;

  const BugType BugMsg{this, "DebugUseAfterLifetimeEnd",
                       "DebugUseAfterLifetimeEnd"};
  using FnCheck = void (DebugUseAfterLifetimeEnd::*)(const CallEvent &Call,
                                                     CheckerContext &C) const;

  const CallDescriptionMap<FnCheck> Callbacks = {
      {{CDM::SimpleFunc, {"clang_analyzer_dumpLifetimeOriginsOf"}},
       &DebugUseAfterLifetimeEnd::analyzerDumpLifetimeOriginsOf},
  };
};

} // namespace

bool DebugUseAfterLifetimeEnd::evalCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  const auto *CE = dyn_cast_if_present<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return false;

  const FnCheck *Handler = Callbacks.lookup(Call);
  if (!Handler)
    return false;

  (this->*(*Handler))(Call, C);
  return true;
}

void DebugUseAfterLifetimeEnd::analyzerDumpLifetimeOriginsOf(
    const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  if (Call.getNumArgs() != 1) {
    if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
      auto BR = std::make_unique<PathSensitiveBugReport>(
          BugMsg,
          "clang_analyzer_dumpLifetimeOriginsOf requires exactly 1 argument",
          N);
      C.emitReport(std::move(BR));
    }
    return;
  }

  SVal ArgSVal = Call.getArgSVal(0);
  const LifetimeSourceSet *SourceSet = State->get<LifetimeBoundMap>(ArgSVal);

  if (!SourceSet)
    return;

  if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
    llvm::SmallVector<std::string> RegionNames =
        to_vector(map_range(llvm::make_pointee_range(*SourceSet),
                            std::mem_fn(&MemRegion::getString)));
    llvm::sort(RegionNames);

    llvm::SmallString<128> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << " Origin " << ArgSVal << " bound to ";
    llvm::interleaveComma(RegionNames, OS);
    C.emitReport(std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N));
  }
}

void ento::registerUseAfterLifetimeEnd(CheckerManager &Mgr) {
  Mgr.registerChecker<UseAfterLifetimeEnd>();
}

bool ento::shouldRegisterUseAfterLifetimeEnd(const CheckerManager &Mgr) {
  return true;
}

void ento::registerDebugUseAfterLifetimeEnd(CheckerManager &Mgr) {
  Mgr.registerChecker<DebugUseAfterLifetimeEnd>();
}

bool ento::shouldRegisterDebugUseAfterLifetimeEnd(const CheckerManager &Mgr) {
  return true;
}
