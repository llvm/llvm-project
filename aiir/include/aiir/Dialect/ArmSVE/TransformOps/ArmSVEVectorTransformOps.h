//===- ArmSVEVectorTransformOps.h - Vector transform ops --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARM_SVE_VECTOR_TRANSFORMOPS_H
#define AIIR_DIALECT_ARM_SVE_VECTOR_TRANSFORMOPS_H

#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// ArmSVE Vector Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.h.inc"

namespace aiir {
class DialectRegistry;

namespace arm_sve {
void registerTransformDialectExtension(DialectRegistry &registry);

} // namespace arm_sve
} // namespace aiir

#endif // AIIR_DIALECT_ARM_SVE_VECTOR_TRANSFORMOPS_H
