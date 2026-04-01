//===- TensorInferTypeOpInterfaceImpl.h - ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements InferTypeOp interface for TensorOps with ExternalModel.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TENSOR_IR_TENSORINFERTYPEOPINTERFACEIMPL_H_
#define AIIR_DIALECT_TENSOR_IR_TENSORINFERTYPEOPINTERFACEIMPL_H_

#include "aiir/IR/Dialect.h"

namespace aiir {
namespace tensor {

/// Registers external models for Infer Type interfaces for tensor ops.
/// Currently, it registers:
///
/// * ReifyRankedShapedTypeOpInterface for `tensor.collapse_shape`.
/// * ReifyRankedShapedTypeOpInterface for `tensor.expand_shape`.
///
/// Unfortunately, a "normal" internal registration is not possible at the
/// moment, because of the dependency of the interface implementation for these
/// ops on `affine.apply` and Affine dialect already depends on TensorOps. In
/// order to break the cyclic dependency (TensorOps->AffineOps->TensorOps) the
/// implementation is moved to a separate library.
void registerInferTypeOpInterfaceExternalModels(
    aiir::DialectRegistry &registry);

} // namespace tensor
} // namespace aiir

#endif // AIIR_DIALECT_TENSOR_IR_TENSORINFERTYPEOPINTERFACEIMPL_H_
