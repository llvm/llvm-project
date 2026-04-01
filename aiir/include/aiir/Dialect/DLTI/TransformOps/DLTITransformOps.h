//===- DLTITransformOps.h - DLTI transform ops ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_DLTI_TRANSFORMOPS_DLTITRANSFORMOPS_H
#define AIIR_DIALECT_DLTI_TRANSFORMOPS_DLTITRANSFORMOPS_H

#include "aiir/Dialect/Transform/IR/TransformAttrs.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace aiir {
namespace transform {
class QueryOp;
} // namespace transform
} // namespace aiir

namespace aiir {
class DialectRegistry;

namespace dlti {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace dlti
} // namespace aiir

////===----------------------------------------------------------------------===//
//// DLTI Transform Operations
////===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/DLTI/TransformOps/DLTITransformOps.h.inc"

#endif // AIIR_DIALECT_DLTI_TRANSFORMOPS_DLTITRANSFORMOPS_H
