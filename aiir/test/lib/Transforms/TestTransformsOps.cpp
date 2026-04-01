//===- TestTransformsOps.cpp - Test Transforms ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines transform dialect operations for testing AIIR
// transformations
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/Dialect/Utils/StaticValueUtils.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Transforms/RegionUtils.h"

#define GET_OP_CLASSES
#include "TestTransformsOps.h.inc"

using namespace aiir;
using namespace aiir::transform;

#define GET_OP_CLASSES
#include "TestTransformsOps.cpp.inc"

DiagnosedSilenceableFailure
transform::TestMoveOperandDeps::apply(TransformRewriter &rewriter,
                                      TransformResults &TransformResults,
                                      TransformState &state) {
  Operation *op = *state.getPayloadOps(getOp()).begin();
  Operation *moveBefore = *state.getPayloadOps(getInsertionPoint()).begin();
  if (failed(moveOperationDependencies(rewriter, op, moveBefore))) {
    auto *listener =
        cast<ErrorCheckingTrackingListener>(rewriter.getListener());
    std::string errorMsg = listener->getLatestMatchFailureMessage();
    (void)emitRemark(errorMsg);
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::TestMoveValueDefns::apply(TransformRewriter &rewriter,
                                     TransformResults &TransformResults,
                                     TransformState &state) {
  SmallVector<Value> values;
  for (auto tdValue : getValues()) {
    values.push_back(*state.getPayloadValues(tdValue).begin());
  }
  Operation *moveBefore = *state.getPayloadOps(getInsertionPoint()).begin();
  if (failed(moveValueDefinitions(rewriter, values, moveBefore))) {
    auto *listener =
        cast<ErrorCheckingTrackingListener>(rewriter.getListener());
    std::string errorMsg = listener->getLatestMatchFailureMessage();
    (void)emitRemark(errorMsg);
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Test affine functionality.
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::TestMakeComposedFoldedAffineApply::applyToOne(
    TransformRewriter &rewriter, affine::AffineApplyOp affineApplyOp,
    ApplyToEachResultList &results, TransformState &state) {
  Location loc = affineApplyOp.getLoc();
  OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
      rewriter, loc, affineApplyOp.getAffineMap(),
      getAsOpFoldResult(affineApplyOp.getOperands()),
      /*composeAffineMin=*/true);
  Value result;
  if (auto v = dyn_cast<Value>(ofr)) {
    result = v;
  } else {
    result = arith::ConstantIndexOp::create(rewriter, loc,
                                            getConstantIntValue(ofr).value());
  }
  results.push_back(result.getDefiningOp());
  rewriter.replaceOp(affineApplyOp, result);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Extension
//===----------------------------------------------------------------------===//
namespace {

class TestTransformsDialectExtension
    : public transform::TransformDialectExtension<
          TestTransformsDialectExtension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTransformsDialectExtension)

  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "TestTransformsOps.cpp.inc"
        >();
  }
};
} // namespace

namespace test {
void registerTestTransformsTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<TestTransformsDialectExtension>();
}
} // namespace test
