//===- Bufferization.h - Bufferization dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATION_H_
#define AIIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATION_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "aiir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "aiir/Interfaces/DestinationStyleOpInterface.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SubsetOpInterface.h"

//===----------------------------------------------------------------------===//
// Bufferization Dialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Bufferization/IR/BufferizationOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Bufferization Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/Bufferization/IR/BufferizationOps.h.inc"

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

namespace aiir {
namespace bufferization {
/// Populate `dynamicDims` with tensor::DimOp / memref::DimOp results for all
/// dynamic dimensions of the given shaped value.
void populateDynamicDimSizes(OpBuilder &b, Location loc, Value shapedValue,
                             SmallVector<Value> &dynamicDims);

/// Try to cast the given ranked MemRef-typed value to the given ranked MemRef
/// type. Insert a reallocation + copy if it cannot be statically guaranteed
/// that a direct cast would be valid.
///
/// E.g., when casting from a ranked MemRef type with dynamic layout to a ranked
/// MemRef type with static layout, it is not statically known whether the cast
/// will succeed or not. Such `memref.cast` ops may fail at runtime. This
/// function never generates such casts and conservatively inserts a copy.
///
/// This function returns `failure()` in case of unsupported casts. E.g., casts
/// with differing element types or memory spaces.
FailureOr<Value> castOrReallocMemRefValue(OpBuilder &b, Value value,
                                          MemRefType type,
                                          const BufferizationOptions &options);

/// Try to fold to_buffer(to_tensor(x)). If x's type and the result type of the
/// to_buffer op are different, a memref.cast is needed.
LogicalResult foldToBufferToTensorPair(RewriterBase &rewriter,
                                       ToBufferOp toBuffer,
                                       const BufferizationOptions &options);

/// Add the canonicalization patterns for bufferization.dealloc to the given
/// pattern set to make them available to other passes (such as
/// BufferDeallocationSimplification).
void populateDeallocOpCanonicalizationPatterns(RewritePatternSet &patterns,
                                               AIIRContext *context);

} // namespace bufferization
} // namespace aiir

#endif // AIIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATION_H_
