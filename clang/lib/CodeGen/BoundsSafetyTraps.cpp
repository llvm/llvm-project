//===- BoundsSafetyTraps.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "BoundsSafetyTraps.h"
#include "BoundsSafetyOptRemarks.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
namespace CodeGen {

llvm::StringRef GetBoundsSafetyTrapMessagePrefix() {
  return "Bounds check failed";
}

llvm::StringRef
GetBoundsSafetyTrapMessageSuffixWithContext(BoundsSafetyTrapKind kind,
                                         BoundsSafetyTrapCtx::Kind TrapCtx) {
  assert(TrapCtx != BoundsSafetyTrapCtx::UNKNOWN);
  switch (kind) {
  case BNS_TRAP_PTR_LT_LOWER_BOUND:
    switch (TrapCtx) {
    case BoundsSafetyTrapCtx::DEREF:
      return "Dereferencing below bounds";
    case BoundsSafetyTrapCtx::CAST:
      return "Pointer below bounds while casting";
    case BoundsSafetyTrapCtx::ASSIGN:
      return "Pointer below bounds while assigning";
    case BoundsSafetyTrapCtx::ADDR_OF_STRUCT_MEMBER:
      return "Pointer to struct below bounds while taking address of struct "
             "member";
    case BoundsSafetyTrapCtx::TERMINATED_BY_FROM_INDEXABLE:
    default:
      llvm_unreachable("Invalid TrapCtx");
    }
  case BNS_TRAP_PTR_GT_UPPER_BOUND:
  case BNS_TRAP_PTR_GE_UPPER_BOUND:
    switch (TrapCtx) {
    case BoundsSafetyTrapCtx::DEREF:
      return "Dereferencing above bounds";
    case BoundsSafetyTrapCtx::CAST:
      return "Pointer above bounds while casting";
    case BoundsSafetyTrapCtx::ASSIGN:
      return "Pointer above bounds while assigning";
    case BoundsSafetyTrapCtx::ADDR_OF_STRUCT_MEMBER:
      return "Pointer to struct above bounds while taking address of struct "
             "member";
    case BoundsSafetyTrapCtx::TERMINATED_BY_FROM_INDEXABLE:
      return "Pointer above bounds while converting __indexable to "
             "__terminated_by";
    default:
      llvm_unreachable("Invalid TrapCtx");
    }
  default:
    llvm_unreachable("Unhandled BoundsSafetyTrapKind");
  }
}

llvm::StringRef GetBoundsSafetyTrapMessageSuffix(BoundsSafetyTrapKind kind,
                                              BoundsSafetyTrapCtx::Kind TrapCtx) {
  switch (kind) {
  case BNS_TRAP_NONE:
    return "";
#define BOUNDS_SAFETY_TRAP_CTX(SUFFIX, ANNOTATION_STR)                             \
  case BNS_TRAP_##SUFFIX:                                                       \
    return GetBoundsSafetyTrapMessageSuffixWithContext(kind, TrapCtx);
#define BOUNDS_SAFETY_TRAP(SUFFIX, ANNOTATION_STR, TRAP_MSG)                       \
  case BNS_TRAP_##SUFFIX:                                                       \
    return TRAP_MSG;
#include "BoundsSafetyTraps.def"
#undef BOUNDS_SAFETY_TRAP
#undef BOUNDS_SAFETY_TRAP_CTX
  }
  llvm_unreachable("Unhandled BoundsSafetyTrapKind");
}

} // namespace CodeGen
} // namespace clang
