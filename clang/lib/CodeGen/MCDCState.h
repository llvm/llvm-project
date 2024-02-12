//===---- MCDCState.h - MC/DC-related types for PGO -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  MC/DC-related types for PGO
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_MCDCSTATE_H
#define LLVM_CLANG_LIB_CODEGEN_MCDCSTATE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"

namespace clang::CodeGen::mcdc {

using ConditionID = llvm::coverage::CounterMappingRegion::MCDCConditionID;
using Parameters = llvm::coverage::CounterMappingRegion::MCDCParameters;

/// Per-Function MC/DC state
struct State {
  unsigned BitmapBytes = 0;
  llvm::DenseMap<const Stmt *, unsigned> BitmapMap;
  llvm::DenseMap<const Stmt *, ConditionID> CondIDMap;
};

} // namespace clang::CodeGen::mcdc

#endif // LLVM_CLANG_LIB_CODEGEN_MCDCSTATE_H
