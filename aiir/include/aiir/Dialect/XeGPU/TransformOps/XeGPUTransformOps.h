//===- XeGPUTransformOps.h - XeGPU transformation ops -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_XEGPU_TRANSFORMOPS_XEGPUTRANSFORMOPS_H
#define AIIR_DIALECT_XEGPU_TRANSFORMOPS_XEGPUTRANSFORMOPS_H

#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/Dialect/Utils/StaticValueUtils.h"

#define GET_OP_CLASSES
#include "aiir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.h.inc"

namespace aiir {
class DialectRegistry;

namespace xegpu {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace xegpu
} // namespace aiir

#endif // AIIR_DIALECT_XEGPU_TRANSFORMOPS_XEGPUTRANSFORMOPS_H
