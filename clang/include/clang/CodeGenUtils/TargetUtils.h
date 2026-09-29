//===--- TargetUtils.h - Shared target-specific AST queries -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the target-specific AST queries that both classic CodeGen
// and CIR CodeGen need.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_TARGETUTILS_H
#define LLVM_CLANG_CODEGENUTILS_TARGETUTILS_H

#include "clang/Basic/BitmaskEnum.h"

#include <cstdint>

namespace clang {
class FunctionDecl;
} // namespace clang

namespace clang::CodeGenUtils {

//===----------------------------------------------------------------------===//
// Arm and AArch64
//===----------------------------------------------------------------------===//

/// The Arm SME ABI issues that can prevent inlining one function into another.
enum class ArmSMEInlinability : uint8_t {
  Ok = 0,
  ErrorCalleeRequiresNewZA = 1 << 0,
  ErrorCalleeRequiresNewZT0 = 1 << 1,
  WarnIncompatibleStreamingModes = 1 << 2,
  ErrorIncompatibleStreamingModes = 1 << 3,

  IncompatibleStreamingModes = WarnIncompatibleStreamingModes |
      ErrorIncompatibleStreamingModes,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/ErrorIncompatibleStreamingModes),
};

/// Determines if there are any Arm SME ABI issues with inlining \p Callee into
/// \p Caller. Returns the issue (if any) in the ArmSMEInlinability bit enum.
ArmSMEInlinability getArmSMEInlinability(const FunctionDecl *Caller,
                                         const FunctionDecl *Callee);

/// Returns whether the Neon builtin \p BuiltinID takes a trailing argument
/// that discriminates the operand type.  This should be kept consistent with
/// the logic in Sema.
/// TODO: Make this return false for SISD builtins.
bool hasExtraNeonArgument(unsigned BuiltinID);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_TARGETUTILS_H
