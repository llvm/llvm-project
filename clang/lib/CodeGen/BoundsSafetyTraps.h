//===- BoundsSafetyTraps.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LIB_CODEGEN_BOUNDS_SAFETY_TRAPS_H
#define LLVM_CLANG_LIB_CODEGEN_BOUNDS_SAFETY_TRAPS_H
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace CodeGen {

enum BoundsSafetyTrapKind {
  BNS_TRAP_NONE, ///< Special value representing no trap.
#define BOUNDS_SAFETY_TRAP_CTX(SUFFIX, ANNOTATION_STR) BNS_TRAP_##SUFFIX,
#define BOUNDS_SAFETY_TRAP(SUFFIX, ANNOTATION_STR, TRAP_MSG) BNS_TRAP_##SUFFIX,
#include "BoundsSafetyTraps.def"

#undef BOUNDS_SAFETY_TRAP
#undef BOUNDS_SAFETY_TRAP_CTX
};

llvm::StringRef GetBoundsSafetyTrapMessagePrefix();

// BoundsSafetyTrapCtx provides an enum and helper methods for describing the
// context where a trap happens (i.e. the operation we are guarding against with
// a trap).
struct BoundsSafetyTrapCtx {
  enum Kind {
    UNKNOWN,                     ///< Unknown
    DEREF,                       ///< Pointer/Array dereference
    ASSIGN,                      ///< Assign
    CAST,                        ///< Cast
    ADDR_OF_STRUCT_MEMBER,       ///< Take address of struct member
    TERMINATED_BY_FROM_INDEXABLE ///< Check during call to
                                 ///< `__unsafe_terminated_by_from_indexable`.
  };
};

llvm::StringRef GetBoundsSafetyTrapMessageSuffix(
    BoundsSafetyTrapKind kind,
    BoundsSafetyTrapCtx::Kind TrapCtx = BoundsSafetyTrapCtx::UNKNOWN);

} // namespace CodeGen
} // namespace clang

#endif
