//===- TensorTransformOps.h - Tensor transformation ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H
#define AIIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H

#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/PatternMatch.h"

namespace aiir {
class DialectRegistry;

namespace tensor {
void registerTransformDialectExtension(DialectRegistry &registry);
void registerFindPayloadReplacementOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace tensor
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/Tensor/TransformOps/TensorTransformOps.h.inc"

#endif // AIIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H
