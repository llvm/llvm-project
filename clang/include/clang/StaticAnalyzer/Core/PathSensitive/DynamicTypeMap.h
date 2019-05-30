//===- DynamicTypeMap.h - Dynamic type map ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file provides APIs for tracking dynamic type information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_DYNAMICTYPEMAP_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_DYNAMICTYPEMAP_H

#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicTypeInfo.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "llvm/ADT/ImmutableMap.h"
#include "clang/AST/Type.h"

namespace clang {
namespace ento {

class MemRegion;

/// The GDM component containing the dynamic type info. This is a map from a
/// symbol to its most likely type.
struct DynamicTypeMap {};

using DynamicTypeMapTy = llvm::ImmutableMap<const MemRegion *, DynamicTypeInfo>;

template <>
struct ProgramStateTrait<DynamicTypeMap>
    : public ProgramStatePartialTrait<DynamicTypeMapTy> {
  static void *GDMIndex();
};

/// Get dynamic type information for a region.
DynamicTypeInfo getDynamicTypeInfo(ProgramStateRef State,
                                   const MemRegion *Reg);

/// Set dynamic type information of the region; return the new state.
ProgramStateRef setDynamicTypeInfo(ProgramStateRef State, const MemRegion *Reg,
                                   DynamicTypeInfo NewTy);

/// Set dynamic type information of the region; return the new state.
inline ProgramStateRef setDynamicTypeInfo(ProgramStateRef State,
                                          const MemRegion *Reg, QualType NewTy,
                                          bool CanBeSubClassed = true) {
  return setDynamicTypeInfo(State, Reg,
                            DynamicTypeInfo(NewTy, CanBeSubClassed));
}

void printDynamicTypeInfoJson(raw_ostream &Out, ProgramStateRef State,
                              const char *NL = "\n", unsigned int Space = 0,
                              bool IsDot = false);

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_DYNAMICTYPEMAP_H
