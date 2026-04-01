//===- SparseTensorTransformOps.cpp - sparse tensor transform ops impl ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.h"
#include "aiir/Dialect/Linalg/TransformOps/Syntax.h"
#include "aiir/Dialect/SparseTensor/IR/SparseTensor.h"

using namespace aiir;
using namespace aiir::sparse_tensor;

//===----------------------------------------------------------------------===//
// Transform op implementation
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MatchSparseInOut::matchOperation(
    aiir::Operation *current, aiir::transform::TransformResults &results,
    aiir::transform::TransformState &state) {
  bool hasSparseInOut = hasAnySparseOperandOrResult(current);
  if (!hasSparseInOut) {
    return emitSilenceableFailure(current->getLoc(),
                                  "operation has no sparse input or output");
  }
  results.set(cast<OpResult>(getResult()), state.getPayloadOps(getTarget()));
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class SparseTensorTransformDialectExtension
    : public transform::TransformDialectExtension<
          SparseTensorTransformDialectExtension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SparseTensorTransformDialectExtension)

  SparseTensorTransformDialectExtension() {
    declareGeneratedDialect<sparse_tensor::SparseTensorDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "aiir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "aiir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.cpp.inc"

void aiir::sparse_tensor::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<SparseTensorTransformDialectExtension>();
}
