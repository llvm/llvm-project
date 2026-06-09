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

REGISTER_SET_WITH_PROGRAMSTATE(LifetimeBoundSet, LifetimeMap)

REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMapVal, const MemRegion *,
                               const MemRegion *)

namespace {
class LifetimeAnnotations : public Checker<check::PostCall, eval::Call> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void analyzerLifetimeBound(const CallEvent &Call, const CallExpr *,
                             CheckerContext &C) const;

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

      if (const MemRegion *ArgValRegion = Arg.getAsRegion()) {
        if (RetValSym)
          State = State->add<LifetimeBoundSet>(
              LifetimeMap{RetValSym, ArgValRegion});
        else if (const MemRegion *RetValRegion = RetVal.getAsRegion())
          State = State->set<LifetimeBoundMapVal>(RetValRegion, ArgValRegion);
      }
    }
  }

  if (const auto *IC = dyn_cast<CXXInstanceCall>(&Call)) {
    if (clang::lifetimes::implicitObjectParamIsLifetimeBound(FD)) {
      if (const MemRegion *AttrRegion = IC->getCXXThisVal().getAsRegion()) {
        if (RetValSym)
          State =
              State->add<LifetimeBoundSet>(LifetimeMap{RetValSym, AttrRegion});
        else if (const MemRegion *RetValRegion = RetVal.getAsRegion())
          State = State->set<LifetimeBoundMapVal>(RetValRegion, AttrRegion);
      }
    }
  }
  C.addTransition(State);
}

void LifetimeAnnotations::printState(raw_ostream &Out, ProgramStateRef State,
                                     const char *NL, const char *Sep) const {
  auto LBMap = State->get<LifetimeBoundSet>();
  auto LBMapVal = State->get<LifetimeBoundMapVal>();

  if (LBMap.isEmpty() && LBMapVal.isEmpty())
    return;

  Out << Sep << "LifetimeBound bindings:" << NL;
  for (auto &&[RetValSym, ArgValRegion] : LBMap) {
    Out << " Origin " << RetValSym << " contains Loan " << ArgValRegion << NL;
  }
  for (auto &&[RetVal, ArgValRegion] : LBMapVal) {
    Out << " Origin " << RetVal << " contains Loan " << ArgValRegion << NL;
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
    auto LBSet = State->get<LifetimeBoundSet>();
    for (const LifetimeMap &Entry : LBSet) {
      if (Entry.SymRefType == ArgSValSym) {
        OS << " Origin " << ArgSValSym << " bound to " << Entry.MemRegType;
        auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N);
        C.emitReport(std::move(BR));
        Str.clear();
      }
    }
  }

  if (ArgValRegion) {
    if (const auto *AttrValLookFor =
            State->get<LifetimeBoundMapVal>(ArgValRegion)) {
      OS << " Origin " << ArgValRegion << " bound to " << *AttrValLookFor;
      auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N);
      C.emitReport(std::move(BR));
      Str.clear();
    }
  }
}

void ento::registerLifetimeAnnotations(CheckerManager &mgr) {
  mgr.registerChecker<LifetimeAnnotations>();
}

bool ento::shouldRegisterLifetimeAnnotations(const CheckerManager &mgr) {
  return true;
}
