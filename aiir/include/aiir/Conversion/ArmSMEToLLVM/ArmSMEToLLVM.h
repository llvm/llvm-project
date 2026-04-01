//===- ArmSMEToLLVM.h - Convert ArmSME to LLVM dialect ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_ARMSMETOLLVM_ARMSMETOLLVM_H_
#define AIIR_CONVERSION_ARMSMETOLLVM_ARMSMETOLLVM_H_

#include <memory>

#include "aiir/Dialect/ArmSME/Transforms/Passes.h"
#include "aiir/Interfaces/FunctionInterfaces.h"

namespace aiir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTARMSMETOLLVM
#include "aiir/Conversion/Passes.h.inc"

/// Create a pass to convert from the ArmSME dialect to LLVM intrinsics.
std::unique_ptr<Pass>
createConvertArmSMEToLLVMPass(bool dumpTileLiveRanges = false);

/// Configure target to convert from the ArmSME dialect to LLVM intrinsics.
void configureArmSMEToLLVMConversionLegality(ConversionTarget &target);

/// Populate the given list with patterns that convert from the ArmSME dialect
/// to LLVM intrinsics.
void populateArmSMEToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);

} // namespace aiir

#endif // AIIR_CONVERSION_ARMSMETOLLVM_ARMSMETOLLVM_H_
