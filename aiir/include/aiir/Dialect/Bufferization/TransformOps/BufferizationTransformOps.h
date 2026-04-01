//===- BufferizationTransformOps.h - Buff. transf. ops ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_BUFFERIZATION_TRANSFORMOPS_BUFFERIZATIONTRANSFORMOPS_H
#define AIIR_DIALECT_BUFFERIZATION_TRANSFORMOPS_BUFFERIZATIONTRANSFORMOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"

namespace aiir {
namespace tensor {
class EmptyOp;
} // namespace tensor
} // namespace aiir

//===----------------------------------------------------------------------===//
// Bufferization Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h.inc"

namespace aiir {
class DialectRegistry;

namespace bufferization {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace bufferization
} // namespace aiir

#endif // AIIR_DIALECT_BUFFERIZATION_TRANSFORMOPS_BUFFERIZATIONTRANSFORMOPS_H
