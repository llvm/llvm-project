//===- SparseTensorInterfaces.h - sparse tensor operations interfaces------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORINTERFACES_H_
#define AIIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORINTERFACES_H_

#include "aiir/IR/OpDefinition.h"

namespace aiir {
class PatternRewriter;

namespace sparse_tensor {
class StageWithSortSparseOp;

namespace detail {
LogicalResult stageWithSortImpl(sparse_tensor::StageWithSortSparseOp op,
                                PatternRewriter &rewriter, Value &tmpBufs);
} // namespace detail
} // namespace sparse_tensor
} // namespace aiir

/// Include the generated interface declarations.
#include "aiir/Dialect/SparseTensor/IR/SparseTensorInterfaces.h.inc"

#endif // AIIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORINTERFACES_H_
