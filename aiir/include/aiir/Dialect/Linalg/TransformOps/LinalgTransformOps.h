//===- LinalgTransformOps.h - Linalg transform ops --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
#define AIIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H

#include "aiir/Dialect/Bufferization/IR/Bufferization.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Transform/IR/TransformAttrs.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/Dialect/Utils/StructuredOpsUtils.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/RegionKindInterface.h"

namespace aiir {
class TilingInterface;
class RewriterBase;

namespace linalg {
class CopyOp;
struct ForallTilingResult;
class GenericOp;
class LinalgOp;
} // namespace linalg

namespace scf {
struct SCFTilingResult;
} // namespace scf

namespace tensor {
class InsertSliceOp;
class PackOp;
class PadOp;
class UnPackOp;
} // namespace tensor

namespace transform {
// Types needed for builders.
struct TileSizesSpec {};
struct NumThreadsSpec {};
} // namespace transform
} // namespace aiir

namespace aiir {
class DialectRegistry;

namespace transform {

/// Implementation of tiling operations using `scf.forall`.
DiagnosedSilenceableFailure
tileToForallOpImpl(RewriterBase &rewriter, transform::TransformState &state,
                   TransformOpInterface transformOp, Operation *target,
                   ArrayRef<OpFoldResult> mixedNumThreads,
                   ArrayRef<OpFoldResult> mixedTileSizes,
                   std::optional<ArrayAttr> mapping,
                   scf::SCFTilingResult &tilingResult);

} // namespace transform
} // namespace aiir

//===----------------------------------------------------------------------===//
// Linalg Transform Operations
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Linalg/TransformOps/LinalgTransformOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/Linalg/TransformOps/LinalgTransformOps.h.inc"

#endif // AIIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
