//===- Passes.h - LLVM Pass Construction and Registration -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H
#define AIIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H

#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/LLVMIR/Transforms/AddComdats.h"
#include "aiir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "aiir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"
#include "aiir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "aiir/Pass/Pass.h"

namespace aiir {

namespace LLVM {

/// Generate the code for registering conversion passes.
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/LLVMIR/Transforms/Passes.h.inc"

} // namespace LLVM
} // namespace aiir

#endif // AIIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H
