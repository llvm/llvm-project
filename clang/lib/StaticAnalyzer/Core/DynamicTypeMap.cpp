//==- DynamicTypeMap.cpp - Dynamic Type Info related APIs ----------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines APIs that track and query dynamic type information. This
//  information can be used to devirtualize calls during the symbolic execution
//  or do type checking.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicTypeMap.h"

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
  if (const TypedRegion *TR = dyn_cast<TypedRegion>(Reg))
    return DynamicTypeInfo(TR->getLocationType(), /*CanBeSubclass=*/false);

  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(Reg)) {
    SymbolRef Sym = SR->getSymbol();
    return DynamicTypeInfo(Sym->getType());
  }

  return DynamicTypeInfo();
}

ProgramStateRef setDynamicTypeInfo(ProgramStateRef State, const MemRegion *Reg,
                                   DynamicTypeInfo NewTy) {
  Reg = Reg->StripCasts();
  ProgramStateRef NewState = State->set<DynamicTypeMap>(Reg, NewTy);
  assert(NewState);
  return NewState;
}

} // namespace ento
} // namespace clang
