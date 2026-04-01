//===- Passes.h - XeGPU Patterns and Passes ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_XEGPU_TRANSFORMS_PASSES_H
#define AIIR_DIALECT_XEGPU_TRANSFORMS_PASSES_H

#include "aiir/Pass/Pass.h"

namespace aiir {

namespace xegpu {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "aiir/Dialect/XeGPU/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/XeGPU/Transforms/Passes.h.inc"

} // namespace xegpu
} // namespace aiir

#endif // AIIR_DIALECT_XEGPU_TRANSFORMS_PASSES_H
