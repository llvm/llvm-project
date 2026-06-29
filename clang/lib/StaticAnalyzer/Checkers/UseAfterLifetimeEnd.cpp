#include "clang/AST/Attr.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Checkers/LifetimeModeling.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
class UseAfterLifetimeEnd
    : public Checker<check::EndFunction, check::DeadSymbols> {
public:
  void reportDanglingSource(const MemRegion *Region, ExplodedNode *N,
                            CheckerContext &C) const;
  void checkReturnedBorrower(SVal Val, ProgramStateRef State,
                             CheckerContext &C) const;
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  const BugType BugMsg{this, "UseAfterLifetimeEnd", "LifetimeBound"};
};

} // namespace

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
  auto SourceSet = lifetimemodeling::getLifetimeSourceSet(State, Val);
  if (!SourceSet.empty()) {
    ExplodedNode *N = nullptr;
    for (const MemRegion *Region : SourceSet) {
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
  ProgramStateRef State =
      lifetimemodeling::removeDeadBindings(C.getState(), SymReaper);
  C.addTransition(State);
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
  auto SourceSet = lifetimemodeling::getLifetimeSourceSet(State, ArgSVal);

  if (SourceSet.empty())
    return;

  if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
    llvm::SmallVector<std::string> RegionNames =
        to_vector(map_range(llvm::make_pointee_range(SourceSet),
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
