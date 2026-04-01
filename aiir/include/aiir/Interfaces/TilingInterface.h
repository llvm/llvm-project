//===- TilingInterface.h - Interface for tiling operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the TilingInterface defined in
// `TilingInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INTERFACES_TILINGINTERFACE_H_
#define AIIR_INTERFACES_TILINGINTERFACE_H_

#include "aiir/Dialect/Utils/StructuredOpsUtils.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Operation.h"
#include "aiir/Interfaces/ViewLikeInterface.h"
#include "aiir/Support/LLVM.h"

namespace aiir {

/// Container for result values of tiling.
/// - `tiledOps` contains operations created by the tiling implementation that
///   are returned to the caller for further transformations.
/// - `tiledValues` contains the tiled value corresponding to the result of the
///   untiled operation.
/// - `generatedSlices` contains the list of slices that are generated during
///   tiling. These slices can be used for fusing producers.
struct TilingResult {
  SmallVector<Operation *> tiledOps;
  SmallVector<Value> tiledValues;
  SmallVector<Operation *> generatedSlices;
};

/// Tiling can be thought of as splitting a dimension into 2 and
/// materializing the outer dimension as a loop:
///
/// op[original] -> op[original / x, x] -> loop[original] { op[x] }
///
/// For parallel dimensions, the split can only happen in one way, with both
/// dimensions being parallel. For reduction dimensions however, there is a
/// choice in how we split the reduction dimension. This enum exposes this
/// choice.
enum class ReductionTilingStrategy {
  // [reduction] -> [reduction1, reduction2]
  // -> loop[reduction1] { [reduction2] }
  FullReduction,
  // [reduction] -> [reduction1, parallel2]
  // -> loop[reduction1] { [parallel2] }; merge[reduction1]
  PartialReductionOuterReduction,
  // [reduction] -> [parallel1, reduction2]
  // -> loop[parallel1] { [reduction2] }; merge[parallel1]
  PartialReductionOuterParallel
};

/// Container for the result of merge operation of tiling.
/// - `mergeOps` contains operations created during the merge.
/// - `replacements` contains the values that represents the result of the
/// merge. These are used as replacements for the original tiled operation.
struct MergeResult {
  SmallVector<Operation *> mergeOps;
  SmallVector<Value> replacements;
};

} // namespace aiir

/// Include the ODS generated interface header files.
#include "aiir/Interfaces/TilingInterface.h.inc"

#endif // AIIR_INTERFACES_TILINGINTERFACE_H_
