//===- Passes.h - Transform dialect pass entry points -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES_H
#define AIIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES_H

#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
class Pass;

namespace transform {
#define GEN_PASS_DECL
#include "aiir/Dialect/Transform/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/Transform/Transforms/Passes.h.inc"
} // namespace transform
} // namespace aiir

#endif // AIIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES_H
