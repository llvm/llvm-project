#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

struct LifetimeMap {
  SymbolRef SymRefType;
  const MemRegion *MemRegType;

  bool operator==(const LifetimeMap &Type) const {
    return std::tie(SymRefType, MemRegType) ==
           std::tie(Type.SymRefType, Type.MemRegType);
  }

  bool operator<(const LifetimeMap &Type) const {
    return std::tie(SymRefType, MemRegType) <
           std::tie(Type.SymRefType, Type.MemRegType);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(SymRefType);
    ID.AddPointer(MemRegType);
  }
};

REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(LifetimeSourceSet, const MemRegion *)
REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMap, SymbolRef, LifetimeSourceSet)

REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMapVal, const MemRegion *, LifetimeSourceSet)

namespace {
class LifetimeAnnotations : public Checker<check::PostCall, eval::Call> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void analyzerLifetimeBound(const CallEvent &Call, const CallExpr *,
                             CheckerContext &C) const;
  ProgramStateRef bindValues(ProgramStateRef State, SymbolRef RetValSym, SVal RetVal, const MemRegion *Source) const;


  const BugType BugMsg{this, "LifetimeAnnotations", "LifetimeBound"};

  using FnCheck = void (LifetimeAnnotations::*)(const CallEvent &Call,
                                                const CallExpr *,
                                                CheckerContext &C) const;

  const CallDescriptionMap<FnCheck> Callbacks = {
      {{CDM::SimpleFunc, {"clang_analyzer_lifetime_bound"}},
       &LifetimeAnnotations::analyzerLifetimeBound},
  };
};

} // namespace

ProgramStateRef LifetimeAnnotations::bindValues(ProgramStateRef State, SymbolRef RetValSym, SVal RetVal, const MemRegion *Source) const {
  LifetimeSourceSet::Factory &F = State->getStateManager().get_context<LifetimeSourceSet>();

  if (RetValSym) {
    const LifetimeSourceSet *LBSet = State->get<LifetimeBoundMap>(RetValSym);
    LifetimeSourceSet Set = LBSet ? *LBSet : F.getEmptySet();
    Set = F.add(Set, Source);
    State = State->set<LifetimeBoundMap>(RetValSym, Set);
  }
  else if (const MemRegion *RetValRegion = RetVal.getAsRegion()) {
    const LifetimeSourceSet *LBValSet = State->get<LifetimeBoundMapVal>(RetValRegion);
    LifetimeSourceSet Set = LBValSet ? *LBValSet : F.getEmptySet();
    Set = F.add(Set, Source);
    State = State->set<LifetimeBoundMapVal>(RetValRegion, Set);
  }
  return State;
}


void LifetimeAnnotations::checkPostCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  const auto *FC = dyn_cast<AnyFunctionCall>(&Call);
  if (!FC)
    return;

  const FunctionDecl *FD = FC->getDecl();
  if (!FD)
    return;

  SVal RetVal = Call.getReturnValue();
  SymbolRef RetValSym = RetVal.getAsSymbol(/*IncludeBaseRegions=*/true);

  for (const ParmVarDecl *PVD : FD->parameters()) {
    if (PVD->hasAttr<LifetimeBoundAttr>()) {
      unsigned Idx = PVD->getFunctionScopeIndex();
      SVal Arg = Call.getArgSVal(Idx);
      if (const MemRegion *ArgValRegion = Arg.getAsRegion())
        State = bindValues(State, RetValSym, RetVal, ArgValRegion);
      }
    }

  if (const auto *IC = dyn_cast<CXXInstanceCall>(&Call)) {
    if (clang::lifetimes::implicitObjectParamIsLifetimeBound(FD)) {
      if (const MemRegion *AttrRegion = IC->getCXXThisVal().getAsRegion()) {
        State = bindValues(State, RetValSym, RetVal, AttrRegion);
      }
    }
  }
  C.addTransition(State);
}

void LifetimeAnnotations::printState(raw_ostream &Out, ProgramStateRef State,
                                     const char *NL, const char *Sep) const {
  auto LBMap = State->get<LifetimeBoundMap>();
  auto LBMapVal = State->get<LifetimeBoundMapVal>();

  if (LBMap.isEmpty() && LBMapVal.isEmpty())
    return;

  Out << Sep << "LifetimeBound bindings:" << NL;
  for (auto &&[RetValSym, ArgValRegion] : LBMap) {
    for (const auto *Region : ArgValRegion)
      Out << " Origin " << RetValSym << " contains Loan " << Region << NL;
  }
  for (auto &&[RetVal, ArgValRegion] : LBMapVal) {
    for (const auto *Region : ArgValRegion)
      Out << " Origin " << RetVal << " contains Loan " << Region << NL;
  }
}

bool LifetimeAnnotations::evalCall(const CallEvent &Call,
                                   CheckerContext &C) const {

  const auto *CE = dyn_cast_if_present<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return false;

  const FnCheck *Handler = Callbacks.lookup(Call);
  if (!Handler)
    return false;

  (this->*(*Handler))(Call, CE, C);
  return true;
}

void LifetimeAnnotations::analyzerLifetimeBound(const CallEvent &Call,
                                                const CallExpr *CE,
                                                CheckerContext &C) const {

  ProgramStateRef State = C.getState();
  unsigned int ArgCount = CE->getNumArgs();
  if (ArgCount != 1)
    return;

  SVal ArgSVal = Call.getArgSVal(0);

  const MemRegion *ArgValRegion = ArgSVal.getAsRegion();
  SymbolRef ArgSValSym = ArgSVal.getAsSymbol(/*IncludeBaseRegions=*/true);

  llvm::SmallString<128> Str;
  llvm::raw_svector_ostream OS(Str);
  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;

  if (ArgSValSym) {
    if (auto *SourceSet = State->get<LifetimeBoundMap>(ArgSValSym)) {
      for (const auto *Region : *SourceSet) {
        OS << " Origin " << ArgSValSym << " bound to " << Region;
        auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N);
        C.emitReport(std::move(BR));
        Str.clear();
      }
    }
  }

  if (ArgValRegion) {
    if (auto *SourceSet = State->get<LifetimeBoundMapVal>(ArgValRegion)) {
      for (const auto *Region : *SourceSet) {
        OS << " Origin " << ArgValRegion << " bound to " << Region;
        auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N);
        C.emitReport(std::move(BR));
        Str.clear();
      }
    }
  }
}

void ento::registerLifetimeAnnotations(CheckerManager &mgr) {
  mgr.registerChecker<LifetimeAnnotations>();
}

bool ento::shouldRegisterLifetimeAnnotations(const CheckerManager &mgr) {
  return true;
}
