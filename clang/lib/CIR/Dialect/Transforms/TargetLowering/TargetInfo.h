//===---- TargetInfo.h - Encapsulate target details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/TargetInfo.h. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETINFO_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETINFO_H

#include "LowerModule.h"
#include "TargetLoweringInfo.h"

namespace mlir {
namespace cir {

/// The AVX ABI level for X86 targets.
enum class X86AVXABILevel {
  None,
  AVX,
  AVX512,
};

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LowerModule &CGM, X86AVXABILevel AVXLevel);

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETINFO_H
