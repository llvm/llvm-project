//===- DynamicTypeMap.cpp - Dynamic Type Info related APIs ----------------===//
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

#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicTypeMap.h"
#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymExpr.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

namespace clang {
namespace ento {

DynamicTypeInfo getDynamicTypeInfo(ProgramStateRef State,
                                   const MemRegion *Reg) {
  Reg = Reg->StripCasts();

  // Look up the dynamic type in the GDM.
  const DynamicTypeInfo *GDMType = State->get<DynamicTypeMap>(Reg);
  if (GDMType)
    return *GDMType;

  // Otherwise, fall back to what we know about the region.
  if (const auto *TR = dyn_cast<TypedRegion>(Reg))
    return DynamicTypeInfo(TR->getLocationType(), /*CanBeSubclass=*/false);

  if (const auto *SR = dyn_cast<SymbolicRegion>(Reg)) {
    SymbolRef Sym = SR->getSymbol();
    return DynamicTypeInfo(Sym->getType());
  }

  return {};
}

ProgramStateRef setDynamicTypeInfo(ProgramStateRef State, const MemRegion *Reg,
                                   DynamicTypeInfo NewTy) {
  Reg = Reg->StripCasts();
  ProgramStateRef NewState = State->set<DynamicTypeMap>(Reg, NewTy);
  assert(NewState);
  return NewState;
}

void printDynamicTypeInfo(ProgramStateRef State, raw_ostream &Out,
                          const char *NL, const char *Sep) {
  bool First = true;
  for (const auto &I : State->get<DynamicTypeMap>()) {
    if (First) {
      Out << NL << "Dynamic types of regions:" << NL;
      First = false;
    }
    const MemRegion *MR = I.first;
    const DynamicTypeInfo &DTI = I.second;
    Out << MR << " : ";
    if (DTI.isValid()) {
      Out << DTI.getType()->getPointeeType().getAsString();
      if (DTI.canBeASubClass()) {
        Out << " (or its subclass)";
      }
    } else {
      Out << "Invalid type info";
    }
    Out << NL;
  }
}

void *ProgramStateTrait<DynamicTypeMap>::GDMIndex() {
  static int index = 0;
  return &index;
}

} // namespace ento
} // namespace clang
