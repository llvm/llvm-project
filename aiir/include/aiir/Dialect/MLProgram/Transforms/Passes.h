//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES_H_
#define AIIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES_H_

#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
namespace ml_program {

#define GEN_PASS_DECL
#include "aiir/Dialect/MLProgram/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/MLProgram/Transforms/Passes.h.inc"

} // namespace ml_program
} // namespace aiir

#endif // AIIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES_H_
