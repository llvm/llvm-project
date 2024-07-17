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
#include "clang/CIR/Target/AArch64.h"
#include "clang/CIR/Target/x86.h"

namespace mlir {
namespace cir {

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LowerModule &CGM,
                               ::cir::X86AVXABILevel AVXLevel);

std::unique_ptr<TargetLoweringInfo>
createAArch64TargetLoweringInfo(LowerModule &CGM,
                                ::cir::AArch64ABIKind AVXLevel);

std::unique_ptr<TargetLoweringInfo>
createSPIRVTargetLoweringInfo(LowerModule &CGM);

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETINFO_H
