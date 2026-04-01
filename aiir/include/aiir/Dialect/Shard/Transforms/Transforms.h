//===- Transforms.h - Shard Transforms --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SHARD_TRANSFORMS_TRANSFORMS_H
#define AIIR_DIALECT_SHARD_TRANSFORMS_TRANSFORMS_H

#include "aiir/Dialect/Shard/IR/ShardOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Value.h"
#include "aiir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace aiir {
class RewritePatternSet;
class SymbolTableCollection;
class DialectRegistry;
class ImplicitLocOpBuilder;
namespace shard {

void populateProcessMultiIndexOpLoweringPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection);
void registerProcessMultiIndexOpLoweringDialects(DialectRegistry &registry);

void populateAllSliceOpLoweringPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection);
void registerAllSliceOpLoweringDialects(DialectRegistry &registry);

void populateAllOpLoweringPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection);
void registerAllOpLoweringDialects(DialectRegistry &registry);

TypedValue<IndexType>
createCollectiveProcessGroupSize(GridOp grid, ArrayRef<GridAxis> axes,
                                 ImplicitLocOpBuilder &builder);

// Get process linear index along the given grid axes.
TypedValue<IndexType>
createProcessLinearIndex(ImplicitLocOpBuilder &builder, StringRef grid,
                         ArrayRef<GridAxis> gridAxes = {});
// Get process linear index from a multi-index along the given grid axes .
TypedValue<IndexType>
createProcessLinearIndex(ImplicitLocOpBuilder &builder, StringRef grid,
                         ValueRange processInGroupMultiIndex,
                         ArrayRef<GridAxis> gridAxes = {});

} // namespace shard
} // namespace aiir

#endif // AIIR_DIALECT_SHARD_TRANSFORMS_TRANSFORMS_H
