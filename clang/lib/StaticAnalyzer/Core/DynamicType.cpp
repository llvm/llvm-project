//===- DynamicType.cpp - Dynamic type related APIs --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines APIs that track and query dynamic type information. This
//  information can be used to devirtualize calls during the symbolic execution
//  or do type checking.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicType.h"
#include "clang/Basic/JsonSupport.h"
#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymExpr.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

/// The GDM component containing the dynamic type info. This is a map from a
/// symbol to its most likely type.
REGISTER_MAP_WITH_PROGRAMSTATE(DynamicTypeMap, const clang::ento::MemRegion *,
                               clang::ento::DynamicTypeInfo)

/// A set factory of dynamic cast informations.
REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(CastSet, clang::ento::DynamicCastInfo)

/// A map from symbols to cast informations.
REGISTER_MAP_WITH_PROGRAMSTATE(DynamicCastMap, const clang::ento::MemRegion *,
                               CastSet)

namespace clang {
namespace ento {

DynamicTypeInfo getDynamicTypeInfo(ProgramStateRef State, const MemRegion *MR) {
  MR = MR->StripCasts();

  // Look up the dynamic type in the GDM.
  if (const DynamicTypeInfo *DTI = State->get<DynamicTypeMap>(MR))
    return *DTI;

  // Otherwise, fall back to what we know about the region.
  if (const auto *TR = dyn_cast<TypedRegion>(MR))
    return DynamicTypeInfo(TR->getLocationType(), /*CanBeSub=*/false);

  if (const auto *SR = dyn_cast<SymbolicRegion>(MR)) {
    SymbolRef Sym = SR->getSymbol();
    return DynamicTypeInfo(Sym->getType());
  }

  return {};
}

const DynamicTypeInfo *getRawDynamicTypeInfo(ProgramStateRef State,
                                             const MemRegion *MR) {
  return State->get<DynamicTypeMap>(MR);
}

const DynamicCastInfo *getDynamicCastInfo(ProgramStateRef State,
                                          const MemRegion *MR,
                                          QualType CastFromTy,
                                          QualType CastToTy) {
  const auto *Lookup = State->get<DynamicCastMap>().lookup(MR);
  if (!Lookup)
    return nullptr;

  for (const DynamicCastInfo &Cast : *Lookup)
    if (Cast.equals(CastFromTy, CastToTy))
      return &Cast;

  return nullptr;
}

ProgramStateRef setDynamicTypeInfo(ProgramStateRef State, const MemRegion *MR,
                                   DynamicTypeInfo NewTy) {
  State = State->set<DynamicTypeMap>(MR->StripCasts(), NewTy);
  assert(State);
  return State;
}

ProgramStateRef setDynamicTypeInfo(ProgramStateRef State, const MemRegion *MR,
                                   QualType NewTy, bool CanBeSubClassed) {
  return setDynamicTypeInfo(State, MR, DynamicTypeInfo(NewTy, CanBeSubClassed));
}

ProgramStateRef setDynamicTypeAndCastInfo(ProgramStateRef State,
                                          const MemRegion *MR,
                                          QualType CastFromTy,
                                          QualType CastToTy,
                                          bool CastSucceeds) {
  if (!MR)
    return State;

  if (CastSucceeds) {
    assert((CastToTy->isAnyPointerType() || CastToTy->isReferenceType()) &&
           "DynamicTypeInfo should always be a pointer.");
    State = State->set<DynamicTypeMap>(MR, CastToTy);
  }

  DynamicCastInfo::CastResult ResultKind =
      CastSucceeds ? DynamicCastInfo::CastResult::Success
                   : DynamicCastInfo::CastResult::Failure;

  CastSet::Factory &F = State->get_context<CastSet>();

  const CastSet *TempSet = State->get<DynamicCastMap>(MR);
  CastSet Set = TempSet ? *TempSet : F.getEmptySet();

  Set = F.add(Set, {CastFromTy, CastToTy, ResultKind});
  State = State->set<DynamicCastMap>(MR, Set);

  assert(State);
  return State;
}

template <typename MapTy>
ProgramStateRef removeDead(ProgramStateRef State, const MapTy &Map,
                           SymbolReaper &SR) {
  for (const auto &Elem : Map)
    if (!SR.isLiveRegion(Elem.first))
      State = State->remove<DynamicCastMap>(Elem.first);

  return State;
}

ProgramStateRef removeDeadTypes(ProgramStateRef State, SymbolReaper &SR) {
  return removeDead(State, State->get<DynamicTypeMap>(), SR);
}

ProgramStateRef removeDeadCasts(ProgramStateRef State, SymbolReaper &SR) {
  return removeDead(State, State->get<DynamicCastMap>(), SR);
}

static void printDynamicTypesJson(raw_ostream &Out, ProgramStateRef State,
                                  const char *NL, unsigned int Space,
                                  bool IsDot) {
  Indent(Out, Space, IsDot) << "\"dynamic_types\": ";

  const DynamicTypeMapTy &Map = State->get<DynamicTypeMap>();
  if (Map.isEmpty()) {
    Out << "null," << NL;
    return;
  }

  ++Space;
  Out << '[' << NL;
  for (DynamicTypeMapTy::iterator I = Map.begin(); I != Map.end(); ++I) {
    const MemRegion *MR = I->first;
    const DynamicTypeInfo &DTI = I->second;
    Indent(Out, Space, IsDot)
        << "{ \"region\": \"" << MR << "\", \"dyn_type\": ";
    if (!DTI.isValid()) {
      Out << "null";
    } else {
      Out << '\"' << DTI.getType()->getPointeeType().getAsString()
          << "\", \"sub_classable\": "
          << (DTI.canBeASubClass() ? "true" : "false");
    }
    Out << " }";

    if (std::next(I) != Map.end())
      Out << ',';
    Out << NL;
  }

  --Space;
  Indent(Out, Space, IsDot) << "]," << NL;
}

static void printDynamicCastsJson(raw_ostream &Out, ProgramStateRef State,
                                  const char *NL, unsigned int Space,
                                  bool IsDot) {
  Indent(Out, Space, IsDot) << "\"dynamic_casts\": ";

  const DynamicCastMapTy &Map = State->get<DynamicCastMap>();
  if (Map.isEmpty()) {
    Out << "null," << NL;
    return;
  }

  ++Space;
  Out << '[' << NL;
  for (DynamicCastMapTy::iterator I = Map.begin(); I != Map.end(); ++I) {
    const MemRegion *MR = I->first;
    const CastSet &Set = I->second;

    Indent(Out, Space, IsDot) << "{ \"region\": \"" << MR << "\", \"casts\": ";
    if (Set.isEmpty()) {
      Out << "null ";
    } else {
      ++Space;
      Out << '[' << NL;
      for (CastSet::iterator SI = Set.begin(); SI != Set.end(); ++SI) {
        Indent(Out, Space, IsDot)
            << "{ \"from\": \"" << SI->from().getAsString() << "\", \"to\": \""
            << SI->to().getAsString() << "\", \"kind\": \""
            << (SI->succeeds() ? "success" : "fail") << "\" }";

        if (std::next(SI) != Set.end())
          Out << ',';
        Out << NL;
      }
      --Space;
      Indent(Out, Space, IsDot) << ']';
    }
    Out << '}';

    if (std::next(I) != Map.end())
      Out << ',';
    Out << NL;
  }

  --Space;
  Indent(Out, Space, IsDot) << "]," << NL;
}

void printDynamicTypeInfoJson(raw_ostream &Out, ProgramStateRef State,
                              const char *NL, unsigned int Space, bool IsDot) {
  printDynamicTypesJson(Out, State, NL, Space, IsDot);
  printDynamicCastsJson(Out, State, NL, Space, IsDot);
}

} // namespace ento
} // namespace clang
