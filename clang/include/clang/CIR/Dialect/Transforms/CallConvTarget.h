//===- CallConvTarget.h - CIR calling-convention target ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CALLCONVTARGET_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CALLCONVTARGET_H

namespace cir {

/// The ABI target whose calling-convention rules drive CallConvLowering.
/// None is the unset state used when the pass runs in classification-attr
/// mode instead of selecting a target.
enum class CallConvTarget { None, Test, X86_64, AMDGPU, SPIRV, AArch64 };

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CALLCONVTARGET_H
