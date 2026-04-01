//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the
// shape transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
#define AIIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_

#include "aiir/Pass/Pass.h"

namespace aiir {
class ConversionTarget;
class ModuleOp;
class TypeConverter;
namespace func {
class FuncOp;
} // namespace func
} // namespace aiir

namespace aiir {

#define GEN_PASS_DECL
#include "aiir/Dialect/Shape/Transforms/Passes.h.inc"

/// Collects a set of patterns to rewrite ops within the Shape dialect.
void populateShapeRewritePatterns(RewritePatternSet &patterns);

// Collects a set of patterns to replace all constraints with passing witnesses.
// This is intended to then allow all ShapeConstraint related ops and data to
// have no effects and allow them to be freely removed such as through
// canonicalization and dead code elimination.
//
// After this pass, no cstr_ operations exist.
void populateRemoveShapeConstraintsPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/Shape/Transforms/Passes.h.inc"

} // namespace aiir

#endif // AIIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
