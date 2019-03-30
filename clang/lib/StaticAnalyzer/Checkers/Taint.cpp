//=== Taint.cpp - Taint tracking and basic propagation rules. ------*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines basic, non-domain-specific mechanisms for tracking tainted values.
//
//===----------------------------------------------------------------------===//

#include "Taint.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"

using namespace clang;
using namespace ento;
using namespace taint;

// Fully tainted symbols.
REGISTER_MAP_WITH_PROGRAMSTATE(TaintMap, SymbolRef, TaintTagType)

// Partially tainted symbols.
REGISTER_MAP_FACTORY_WITH_PROGRAMSTATE(TaintedSubRegions, const SubRegion *,
                                       TaintTagType);
REGISTER_MAP_WITH_PROGRAMSTATE(DerivedSymTaint, SymbolRef, TaintedSubRegions)

void taint::printTaint(ProgramStateRef State, raw_ostream &Out, const char *NL,
                       const char *Sep) {
  TaintMapTy TM = State->get<TaintMap>();

  if (!TM.isEmpty())
    Out << "Tainted symbols:" << NL;

  for (const auto &I : TM)
    Out << I.first << " : " << I.second << NL;
}

void dumpTaint(ProgramStateRef State) {
  printTaint(State, llvm::errs());
}

ProgramStateRef taint::addTaint(ProgramStateRef State, const Stmt *S,
                                const LocationContext *LCtx,
                                TaintTagType Kind) {
  return addTaint(State, State->getSVal(S, LCtx), Kind);
}

ProgramStateRef taint::addTaint(ProgramStateRef State, SVal V,
                                TaintTagType Kind) {
  SymbolRef Sym = V.getAsSymbol();
  if (Sym)
    return addTaint(State, Sym, Kind);

  // If the SVal represents a structure, try to mass-taint all values within the
  // structure. For now it only works efficiently on lazy compound values that
  // were conjured during a conservative evaluation of a function - either as
  // return values of functions that return structures or arrays by value, or as
  // values of structures or arrays passed into the function by reference,
  // directly or through pointer aliasing. Such lazy compound values are
  // characterized by having exactly one binding in their captured store within
  // their parent region, which is a conjured symbol default-bound to the base
  // region of the parent region.
  if (auto LCV = V.getAs<nonloc::LazyCompoundVal>()) {
    if (Optional<SVal> binding =
            State->getStateManager().getStoreManager()
                                    .getDefaultBinding(*LCV)) {
      if (SymbolRef Sym = binding->getAsSymbol())
        return addPartialTaint(State, Sym, LCV->getRegion(), Kind);
    }
  }

  const MemRegion *R = V.getAsRegion();
  return addTaint(State, R, Kind);
}

ProgramStateRef taint::addTaint(ProgramStateRef State, const MemRegion *R,
                                TaintTagType Kind) {
  if (const SymbolicRegion *SR = dyn_cast_or_null<SymbolicRegion>(R))
    return addTaint(State, SR->getSymbol(), Kind);
  return State;
}

ProgramStateRef taint::addTaint(ProgramStateRef State, SymbolRef Sym,
                                TaintTagType Kind) {
  // If this is a symbol cast, remove the cast before adding the taint. Taint
  // is cast agnostic.
  while (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym))
    Sym = SC->getOperand();

  ProgramStateRef NewState = State->set<TaintMap>(Sym, Kind);
  assert(NewState);
  return NewState;
}

ProgramStateRef taint::addPartialTaint(ProgramStateRef State,
                                       SymbolRef ParentSym,
                                       const SubRegion *SubRegion,
                                       TaintTagType Kind) {
  // Ignore partial taint if the entire parent symbol is already tainted.
  if (const TaintTagType *T = State->get<TaintMap>(ParentSym))
    if (*T == Kind)
      return State;

  // Partial taint applies if only a portion of the symbol is tainted.
  if (SubRegion == SubRegion->getBaseRegion())
    return addTaint(State, ParentSym, Kind);

  const TaintedSubRegions *SavedRegs = State->get<DerivedSymTaint>(ParentSym);
  TaintedSubRegions::Factory &F = State->get_context<TaintedSubRegions>();
  TaintedSubRegions Regs = SavedRegs ? *SavedRegs : F.getEmptyMap();

  Regs = F.add(Regs, SubRegion, Kind);
  ProgramStateRef NewState = State->set<DerivedSymTaint>(ParentSym, Regs);
  assert(NewState);
  return NewState;
}

bool taint::isTainted(ProgramStateRef State, const Stmt *S,
                      const LocationContext *LCtx, TaintTagType Kind) {
  SVal val = State->getSVal(S, LCtx);
  return isTainted(State, val, Kind);
}

bool taint::isTainted(ProgramStateRef State, SVal V, TaintTagType Kind) {
  if (const SymExpr *Sym = V.getAsSymExpr())
    return isTainted(State, Sym, Kind);
  if (const MemRegion *Reg = V.getAsRegion())
    return isTainted(State, Reg, Kind);
  return false;
}

bool taint::isTainted(ProgramStateRef State, const MemRegion *Reg,
                      TaintTagType K) {
  if (!Reg)
    return false;

  // Element region (array element) is tainted if either the base or the offset
  // are tainted.
  if (const ElementRegion *ER = dyn_cast<ElementRegion>(Reg))
    return isTainted(State, ER->getSuperRegion(), K) ||
           isTainted(State, ER->getIndex(), K);

  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(Reg))
    return isTainted(State, SR->getSymbol(), K);

  if (const SubRegion *ER = dyn_cast<SubRegion>(Reg))
    return isTainted(State, ER->getSuperRegion(), K);

  return false;
}

bool taint::isTainted(ProgramStateRef State, SymbolRef Sym, TaintTagType Kind) {
  if (!Sym)
    return false;

  // Traverse all the symbols this symbol depends on to see if any are tainted.
  for (SymExpr::symbol_iterator SI = Sym->symbol_begin(),
                                SE = Sym->symbol_end(); SI != SE; ++SI) {
    if (!isa<SymbolData>(*SI))
      continue;

    if (const TaintTagType *Tag = State->get<TaintMap>(*SI)) {
      if (*Tag == Kind)
        return true;
    }

    if (const auto *SD = dyn_cast<SymbolDerived>(*SI)) {
      // If this is a SymbolDerived with a tainted parent, it's also tainted.
      if (isTainted(State, SD->getParentSymbol(), Kind))
        return true;

      // If this is a SymbolDerived with the same parent symbol as another
      // tainted SymbolDerived and a region that's a sub-region of that tainted
      // symbol, it's also tainted.
      if (const TaintedSubRegions *Regs =
              State->get<DerivedSymTaint>(SD->getParentSymbol())) {
        const TypedValueRegion *R = SD->getRegion();
        for (auto I : *Regs) {
          // FIXME: The logic to identify tainted regions could be more
          // complete. For example, this would not currently identify
          // overlapping fields in a union as tainted. To identify this we can
          // check for overlapping/nested byte offsets.
          if (Kind == I.second && R->isSubRegionOf(I.first))
            return true;
        }
      }
    }

    // If memory region is tainted, data is also tainted.
    if (const auto *SRV = dyn_cast<SymbolRegionValue>(*SI)) {
      if (isTainted(State, SRV->getRegion(), Kind))
        return true;
    }

    // If this is a SymbolCast from a tainted value, it's also tainted.
    if (const auto *SC = dyn_cast<SymbolCast>(*SI)) {
      if (isTainted(State, SC->getOperand(), Kind))
        return true;
    }
  }

  return false;
}

std::shared_ptr<PathDiagnosticPiece>
TaintBugVisitor::VisitNode(const ExplodedNode *N, BugReporterContext &BRC,
                           BugReport &BR) {

  // Find the ExplodedNode where the taint was first introduced
  if (!isTainted(N->getState(), V) ||
      isTainted(N->getFirstPred()->getState(), V))
    return nullptr;

  const Stmt *S = PathDiagnosticLocation::getStmt(N);
  if (!S)
    return nullptr;

  const LocationContext *NCtx = N->getLocationContext();
  PathDiagnosticLocation L =
      PathDiagnosticLocation::createBegin(S, BRC.getSourceManager(), NCtx);
  if (!L.isValid() || !L.asLocation().isValid())
    return nullptr;

  return std::make_shared<PathDiagnosticEventPiece>(L, "Taint originated here");
}
