//===- BoundsSafetyOptRemarks.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "BoundsSafetyOptRemarks.h"
#include "BoundsSafetyTraps.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
namespace CodeGen {

llvm::StringRef GetBoundsSafetyOptRemarkString(BoundsSafetyOptRemarkKind kind) {
  switch (kind) {
  case BNS_OR_NONE:
    return "";
#define BOUNDS_SAFETY_OR(SUFFIX, ANNOTATION_STR)                                   \
  case BNS_OR_##SUFFIX:                                                         \
    return ANNOTATION_STR;
#include "BoundsSafetyOptRemarks.def"
#undef BOUNDS_SAFETY_OR
  }
  llvm_unreachable("Unhandled BoundsSafetyOptRemarkKind");
}

BoundsSafetyOptRemarkKind GetBoundsSafetyOptRemarkForTrap(BoundsSafetyTrapKind kind) {
  switch (kind) {
  case BNS_TRAP_NONE:
    return BNS_OR_NONE;
#define BOUNDS_SAFETY_TRAP_CTX(SUFFIX, ANNOTATION_STR)                             \
  case BNS_TRAP_##SUFFIX:                                                       \
    return BNS_OR_##SUFFIX;
#define BOUNDS_SAFETY_TRAP(SUFFIX, ANNOTATION_STR, TRAP_MSG)                       \
  case BNS_TRAP_##SUFFIX:                                                       \
    return BNS_OR_##SUFFIX;
#include "BoundsSafetyTraps.def"
#undef BOUNDS_SAFETY_TRAP
#undef BOUNDS_SAFETY_TRAP_CTX
  }
  llvm_unreachable("Unhandled BoundsSafetyTrapKind");
}

} // namespace CodeGen
} // namespace clang
