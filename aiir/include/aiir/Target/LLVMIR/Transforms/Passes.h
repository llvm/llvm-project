//===- Passes.h - LLVM Target Pass Construction and Registration ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_TRANSFORMS_PASSES_H
#define AIIR_TARGET_LLVMIR_TRANSFORMS_PASSES_H

#include "aiir/Pass/Pass.h"

namespace aiir {
namespace LLVM {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "aiir/Target/LLVMIR/Transforms/Passes.h.inc"

void registerTargetLLVMPasses();

} // namespace LLVM
} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_TRANSFORMS_PASSES_H
