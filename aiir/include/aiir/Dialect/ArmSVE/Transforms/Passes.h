//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_H
#define AIIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_H

#include "aiir/Conversion/LLVMCommon/TypeConverter.h"
#include "aiir/Pass/Pass.h"

namespace aiir::arm_sve {

#define GEN_PASS_DECL
#include "aiir/Dialect/ArmSVE/Transforms/Passes.h.inc"

/// Pass to legalize Arm SVE vector storage.
std::unique_ptr<Pass> createLegalizeVectorStoragePass();

/// Collect a set of patterns to legalize Arm SVE vector storage.
void populateLegalizeVectorStoragePatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/ArmSVE/Transforms/Passes.h.inc"

} // namespace aiir::arm_sve

#endif // AIIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_H
