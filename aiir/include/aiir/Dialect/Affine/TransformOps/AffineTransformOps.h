//===- AffineTransformOps.h - Affine transformation ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_AFFINE_TRANSFORMOPS_AFFINETRANSFORMOPS_H
#define AIIR_DIALECT_AFFINE_TRANSFORMOPS_AFFINETRANSFORMOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"

namespace aiir {
namespace func {
class FuncOp;
} // namespace func
namespace affine {
class AffineForOp;
} // namespace affine
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/Affine/TransformOps/AffineTransformOps.h.inc"

namespace aiir {
class DialectRegistry;

namespace affine {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace affine
} // namespace aiir

#endif // AIIR_DIALECT_AFFINE_TRANSFORMOPS_AFFINETRANSFORMOPS_H
