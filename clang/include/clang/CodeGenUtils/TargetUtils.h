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
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace llvm {
class Triple;
} // namespace llvm

namespace clang {
class Decl;
class FunctionDecl;
class QualType;
class TargetInfo;
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

/// Helper method to check if the underlying ABI is AAPCS
bool isAAPCS(const TargetInfo &TargetInfo);

//===----------------------------------------------------------------------===//
// X86
//===----------------------------------------------------------------------===//
enum class MSVCIntrin {
  _BitScanForward,
  _BitScanReverse,
  _InterlockedAnd,
  _InterlockedCompareExchange,
  _InterlockedDecrement,
  _InterlockedExchange,
  _InterlockedExchangeAdd,
  _InterlockedExchangeSub,
  _InterlockedIncrement,
  _InterlockedOr,
  _InterlockedXor,
  _InterlockedExchangeAdd_acq,
  _InterlockedExchangeAdd_rel,
  _InterlockedExchangeAdd_nf,
  _InterlockedExchange_acq,
  _InterlockedExchange_rel,
  _InterlockedExchange_nf,
  _InterlockedCompareExchange_acq,
  _InterlockedCompareExchange_rel,
  _InterlockedCompareExchange_nf,
  _InterlockedCompareExchange128,
  _InterlockedCompareExchange128_acq,
  _InterlockedCompareExchange128_rel,
  _InterlockedCompareExchange128_nf,
  _InterlockedOr_acq,
  _InterlockedOr_rel,
  _InterlockedOr_nf,
  _InterlockedXor_acq,
  _InterlockedXor_rel,
  _InterlockedXor_nf,
  _InterlockedAnd_acq,
  _InterlockedAnd_rel,
  _InterlockedAnd_nf,
  _InterlockedIncrement_acq,
  _InterlockedIncrement_rel,
  _InterlockedIncrement_nf,
  _InterlockedDecrement_acq,
  _InterlockedDecrement_rel,
  _InterlockedDecrement_nf,
  __fastfail,
};

std::optional<MSVCIntrin>
translateX86ToMsvcIntrin(unsigned BuiltinID);
  
// Library builtins (PowerPC64 and AIX)
//===----------------------------------------------------------------------===//

/// Returns the library function name that should be called for the library
/// builtin \p BuiltinID when the target's library does not use the name derived
/// from the builtin (e.g. PPC64 IEEE 128-bit long double libcalls, or AIX with
/// 64-bit long double), or an empty StringRef if there is no such override.
llvm::StringRef getTargetBuiltinLibFunctionName(unsigned BuiltinID,
                                                const TargetInfo &Target);

//===----------------------------------------------------------------------===//
// AMDGPU
//===----------------------------------------------------------------------===//

/// Returns whether \p D must be given protected visibility on AMDGPU.
/// \p HasHiddenVisibility is whether the emitted global currently has hidden
/// visibility.
bool requiresAMDGPUProtectedVisibility(const Decl *D, bool HasHiddenVisibility);

//===----------------------------------------------------------------------===//
// SPIR-V
//===----------------------------------------------------------------------===//

/// Return true if a SPIR(-V) null pointer of type \p QT must be materialized
/// as an addrspacecast from a generic null pointer, as the null bit pattern in
/// non-generic address spaces is unspecified.
bool spirNullPointerNeedsGenericCast(QualType QT, const llvm::Triple &Triple);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_TARGETUTILS_H
