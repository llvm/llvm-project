//===- Passes.h - MemRef Patterns and Passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on MemRef operations.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
#define AIIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H

#include "aiir/Pass/Pass.h"

namespace aiir {

class AffineDialect;
class ModuleOp;

namespace func {
namespace arith {
class ArithDialect;
} // namespace arith
class FuncDialect;
} // namespace func
namespace scf {
class SCFDialect;
} // namespace scf
namespace tensor {
class TensorDialect;
} // namespace tensor
namespace vector {
class VectorDialect;
} // namespace vector

namespace memref {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "aiir/Dialect/MemRef/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/MemRef/Transforms/Passes.h.inc"

} // namespace memref
} // namespace aiir

#endif // AIIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
