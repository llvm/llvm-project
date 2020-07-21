// SmartPtrModeling.cpp - Model behavior of C++ smart pointers - C++ ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a checker that models various aspects of
// C++ smart pointer behavior.
//
//===----------------------------------------------------------------------===//

#include "Move.h"
#include "SmartPtr.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymExpr.h"

using namespace clang;
using namespace ento;

namespace {
class SmartPtrModeling
    : public Checker<eval::Call, check::DeadSymbols, check::RegionChanges> {

  bool isNullAfterMoveMethod(const CallEvent &Call) const;

public:
  // Whether the checker should model for null dereferences of smart pointers.
  DefaultBool ModelSmartPtrDereference;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  ProgramStateRef
  checkRegionChanges(ProgramStateRef State,
                     const InvalidatedSymbols *Invalidated,
                     ArrayRef<const MemRegion *> ExplicitRegions,
                     ArrayRef<const MemRegion *> Regions,
                     const LocationContext *LCtx, const CallEvent *Call) const;

private:
  ProgramStateRef updateTrackedRegion(const CallEvent &Call, CheckerContext &C,
                                      const MemRegion *ThisValRegion) const;
  void handleReset(const CallEvent &Call, CheckerContext &C) const;
  void handleRelease(const CallEvent &Call, CheckerContext &C) const;
  void handleSwap(const CallEvent &Call, CheckerContext &C) const;

  using SmartPtrMethodHandlerFn =
      void (SmartPtrModeling::*)(const CallEvent &Call, CheckerContext &) const;
  CallDescriptionMap<SmartPtrMethodHandlerFn> SmartPtrMethodHandlers{
      {{"reset"}, &SmartPtrModeling::handleReset},
      {{"release"}, &SmartPtrModeling::handleRelease},
      {{"swap", 1}, &SmartPtrModeling::handleSwap}};
};
} // end of anonymous namespace

REGISTER_MAP_WITH_PROGRAMSTATE(TrackedRegionMap, const MemRegion *, SVal)

// Define the inter-checker API.
namespace clang {
namespace ento {
namespace smartptr {
bool isStdSmartPtrCall(const CallEvent &Call) {
  const auto *MethodDecl = dyn_cast_or_null<CXXMethodDecl>(Call.getDecl());
  if (!MethodDecl || !MethodDecl->getParent())
    return false;

  const auto *RecordDecl = MethodDecl->getParent();
  if (!RecordDecl || !RecordDecl->getDeclContext()->isStdNamespace())
    return false;

  if (RecordDecl->getDeclName().isIdentifier()) {
    StringRef Name = RecordDecl->getName();
    return Name == "shared_ptr" || Name == "unique_ptr" || Name == "weak_ptr";
  }
  return false;
}

bool isNullSmartPtr(const ProgramStateRef State, const MemRegion *ThisRegion) {
  const auto *InnerPointVal = State->get<TrackedRegionMap>(ThisRegion);
  return InnerPointVal && InnerPointVal->isZeroConstant();
}
} // namespace smartptr
} // namespace ento
} // namespace clang

// If a region is removed all of the subregions need to be removed too.
static TrackedRegionMapTy
removeTrackedSubregions(TrackedRegionMapTy RegionMap,
                        TrackedRegionMapTy::Factory &RegionMapFactory,
                        const MemRegion *Region) {
  if (!Region)
    return RegionMap;
  for (const auto &E : RegionMap) {
    if (E.first->isSubRegionOf(Region))
      RegionMap = RegionMapFactory.remove(RegionMap, E.first);
  }
  return RegionMap;
}

static ProgramStateRef updateSwappedRegion(ProgramStateRef State,
                                           const MemRegion *Region,
                                           const SVal *RegionInnerPointerVal) {
  if (RegionInnerPointerVal) {
    State = State->set<TrackedRegionMap>(Region, *RegionInnerPointerVal);
  } else {
    State = State->remove<TrackedRegionMap>(Region);
  }
  return State;
}

bool SmartPtrModeling::isNullAfterMoveMethod(const CallEvent &Call) const {
  // TODO: Update CallDescription to support anonymous calls?
  // TODO: Handle other methods, such as .get() or .release().
  // But once we do, we'd need a visitor to explain null dereferences
  // that are found via such modeling.
  const auto *CD = dyn_cast_or_null<CXXConversionDecl>(Call.getDecl());
  return CD && CD->getConversionType()->isBooleanType();
}

bool SmartPtrModeling::evalCall(const CallEvent &Call,
                                CheckerContext &C) const {

  if (!smartptr::isStdSmartPtrCall(Call))
    return false;

  if (isNullAfterMoveMethod(Call)) {
    ProgramStateRef State = C.getState();
    const MemRegion *ThisR =
        cast<CXXInstanceCall>(&Call)->getCXXThisVal().getAsRegion();

    if (!move::isMovedFrom(State, ThisR)) {
      // TODO: Model this case as well. At least, avoid invalidation of
      // globals.
      return false;
    }

    // TODO: Add a note to bug reports describing this decision.
    C.addTransition(
        State->BindExpr(Call.getOriginExpr(), C.getLocationContext(),
                        C.getSValBuilder().makeZeroVal(Call.getResultType())));
    return true;
  }

  if (!ModelSmartPtrDereference)
    return false;

  if (const auto *CC = dyn_cast<CXXConstructorCall>(&Call)) {
    if (CC->getDecl()->isCopyOrMoveConstructor())
      return false;

    const MemRegion *ThisValRegion = CC->getCXXThisVal().getAsRegion();
    if (!ThisValRegion)
      return false;

    auto State = updateTrackedRegion(Call, C, ThisValRegion);
    C.addTransition(State);
    return true;
  }

  const SmartPtrMethodHandlerFn *Handler = SmartPtrMethodHandlers.lookup(Call);
  if (!Handler)
    return false;
  (this->**Handler)(Call, C);

  return C.isDifferent();
}

void SmartPtrModeling::checkDeadSymbols(SymbolReaper &SymReaper,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  // Clean up dead regions from the region map.
  TrackedRegionMapTy TrackedRegions = State->get<TrackedRegionMap>();
  for (auto E : TrackedRegions) {
    const MemRegion *Region = E.first;
    bool IsRegDead = !SymReaper.isLiveRegion(Region);

    if (IsRegDead)
      State = State->remove<TrackedRegionMap>(Region);
  }
  C.addTransition(State);
}

ProgramStateRef SmartPtrModeling::checkRegionChanges(
    ProgramStateRef State, const InvalidatedSymbols *Invalidated,
    ArrayRef<const MemRegion *> ExplicitRegions,
    ArrayRef<const MemRegion *> Regions, const LocationContext *LCtx,
    const CallEvent *Call) const {
  TrackedRegionMapTy RegionMap = State->get<TrackedRegionMap>();
  TrackedRegionMapTy::Factory &RegionMapFactory =
      State->get_context<TrackedRegionMap>();
  for (const auto *Region : Regions)
    RegionMap = removeTrackedSubregions(RegionMap, RegionMapFactory,
                                        Region->getBaseRegion());
  return State->set<TrackedRegionMap>(RegionMap);
}

void SmartPtrModeling::handleReset(const CallEvent &Call,
                                   CheckerContext &C) const {
  const auto *IC = dyn_cast<CXXInstanceCall>(&Call);
  if (!IC)
    return;

  const MemRegion *ThisValRegion = IC->getCXXThisVal().getAsRegion();
  if (!ThisValRegion)
    return;
  auto State = updateTrackedRegion(Call, C, ThisValRegion);
  C.addTransition(State);
  // TODO: Make sure to ivalidate the region in the Store if we don't have
  // time to model all methods.
}

void SmartPtrModeling::handleRelease(const CallEvent &Call,
                                     CheckerContext &C) const {
  const auto *IC = dyn_cast<CXXInstanceCall>(&Call);
  if (!IC)
    return;

  const MemRegion *ThisValRegion = IC->getCXXThisVal().getAsRegion();
  if (!ThisValRegion)
    return;

  auto State = updateTrackedRegion(Call, C, ThisValRegion);

  const auto *InnerPointVal = State->get<TrackedRegionMap>(ThisValRegion);
  if (InnerPointVal) {
    State = State->BindExpr(Call.getOriginExpr(), C.getLocationContext(),
                            *InnerPointVal);
  }
  C.addTransition(State);
  // TODO: Add support to enable MallocChecker to start tracking the raw
  // pointer.
}

void SmartPtrModeling::handleSwap(const CallEvent &Call,
                                  CheckerContext &C) const {
  // To model unique_ptr::swap() method.
  const auto *IC = dyn_cast<CXXInstanceCall>(&Call);
  if (!IC)
    return;

  const MemRegion *ThisRegion = IC->getCXXThisVal().getAsRegion();
  if (!ThisRegion)
    return;

  const auto *ArgRegion = Call.getArgSVal(0).getAsRegion();
  if (!ArgRegion)
    return;

  auto State = C.getState();
  const auto *ThisRegionInnerPointerVal =
      State->get<TrackedRegionMap>(ThisRegion);
  const auto *ArgRegionInnerPointerVal =
      State->get<TrackedRegionMap>(ArgRegion);

  // Swap the tracked region values.
  State = updateSwappedRegion(State, ThisRegion, ArgRegionInnerPointerVal);
  State = updateSwappedRegion(State, ArgRegion, ThisRegionInnerPointerVal);

  C.addTransition(State);
}

ProgramStateRef
SmartPtrModeling::updateTrackedRegion(const CallEvent &Call, CheckerContext &C,
                                      const MemRegion *ThisValRegion) const {
  // TODO: Refactor and clean up handling too many things.
  ProgramStateRef State = C.getState();
  auto NumArgs = Call.getNumArgs();

  if (NumArgs == 0) {
    auto NullSVal = C.getSValBuilder().makeNull();
    State = State->set<TrackedRegionMap>(ThisValRegion, NullSVal);
  } else if (NumArgs == 1) {
    auto ArgVal = Call.getArgSVal(0);
    assert(Call.getArgExpr(0)->getType()->isPointerType() &&
           "Adding a non pointer value to TrackedRegionMap");
    State = State->set<TrackedRegionMap>(ThisValRegion, ArgVal);
  }

  return State;
}

void ento::registerSmartPtrModeling(CheckerManager &Mgr) {
  auto *Checker = Mgr.registerChecker<SmartPtrModeling>();
  Checker->ModelSmartPtrDereference =
      Mgr.getAnalyzerOptions().getCheckerBooleanOption(
          Checker, "ModelSmartPtrDereference");
}

bool ento::shouldRegisterSmartPtrModeling(const CheckerManager &mgr) {
  const LangOptions &LO = mgr.getLangOpts();
  return LO.CPlusPlus;
}
