//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LINALG_PASSES_H_
#define AIIR_DIALECT_LINALG_PASSES_H_

#include "aiir/Dialect/Linalg/Transforms/Transforms.h"
#include "aiir/Dialect/Linalg/Utils/Utils.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
namespace func {
class FuncOp;
} // namespace func

#define GEN_PASS_DECL
#include "aiir/Dialect/Linalg/Passes.h.inc" // IWYU pragma: keep

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/Linalg/Passes.h.inc"

} // namespace aiir

#endif // AIIR_DIALECT_LINALG_PASSES_H_
