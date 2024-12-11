//===- BoundsSafetyOptRemarks.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LIB_CODEGEN_BOUNDS_SAFETY_OPT_REMARKS_H
#define LLVM_CLANG_LIB_CODEGEN_BOUNDS_SAFETY_OPT_REMARKS_H
#include "BoundsSafetyTraps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instruction.h"

namespace clang {
namespace CodeGen {

enum BoundsSafetyOptRemarkKind {
  BNS_OR_NONE, ///< Special value representing no opt-remark.
#define BOUNDS_SAFETY_OR(SUFFIX, ANNOTATION_STR) BNS_OR_##SUFFIX,
#include "BoundsSafetyOptRemarks.def"
#undef BOUNDS_SAFETY_OR
};

llvm::StringRef GetBoundsSafetyOptRemarkString(BoundsSafetyOptRemarkKind kind);
BoundsSafetyOptRemarkKind GetBoundsSafetyOptRemarkForTrap(BoundsSafetyTrapKind kind);

} // namespace CodeGen
} // namespace clang

#endif
