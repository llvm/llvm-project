//===- MemRefTransformOps.h - MemRef transformation ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MEMREF_TRANSFORMOPS_MEMREFTRANSFORMOPS_H
#define AIIR_DIALECT_MEMREF_TRANSFORMOPS_MEMREFTRANSFORMOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"

namespace aiir {
namespace memref {
class AllocOp;
} // namespace memref
namespace transform {
class OperationType;
} // namespace transform
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/MemRef/TransformOps/MemRefTransformOps.h.inc"

namespace aiir {
class DialectRegistry;

namespace memref {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace memref
} // namespace aiir

#endif // AIIR_DIALECT_MEMREF_TRANSFORMOPS_MEMREFTRANSFORMOPS_H
