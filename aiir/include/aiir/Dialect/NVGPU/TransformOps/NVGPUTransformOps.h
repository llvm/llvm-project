//===- NVGPUTransformOps.h - NVGPU transform ops ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_NVGPU_TRANSFORMOPS_NVGPUTRANSFORMOPS_H
#define AIIR_DIALECT_NVGPU_TRANSFORMOPS_NVGPUTRANSFORMOPS_H

#include "aiir/Dialect/Transform/IR/TransformAttrs.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/RegionKindInterface.h"

namespace aiir {
namespace transform {
class TransformHandleTypeInterface;
} // namespace transform
} // namespace aiir

namespace aiir {
class DialectRegistry;

namespace linalg {
class LinalgOp;
} // namespace linalg

namespace scf {
class ForOp;
} // namespace scf

namespace nvgpu {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace nvgpu
} // namespace aiir

//===----------------------------------------------------------------------===//
// NVGPU Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h.inc"

#endif // AIIR_DIALECT_NVGPU_TRANSFORMOPS_NVGPUTRANSFORMOPS_H
