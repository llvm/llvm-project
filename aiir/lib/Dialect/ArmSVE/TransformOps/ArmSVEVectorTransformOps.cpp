//===- ArmSVEVectorTransformOps.cpp - Implementation transform ops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.h"

#include "aiir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "aiir/Dialect/ArmSVE/Transforms/Transforms.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"

using namespace aiir;

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyArmSVELowerContractionToI8MMPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  aiir::populateLowerContractionToSVEI8MMPatterns(patterns);
}

void transform::ApplyArmSVELowerContractionToBFMMLAPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  aiir::populateLowerContractionToSVEBFMMLAPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class ArmSVEVectorTransformDialectExtension
    : public transform::TransformDialectExtension<
          ArmSVEVectorTransformDialectExtension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ArmSVEVectorTransformDialectExtension)

  ArmSVEVectorTransformDialectExtension() {
    declareGeneratedDialect<arm_sve::ArmSVEDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "aiir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "aiir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.cpp.inc"

void aiir::arm_sve::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<ArmSVEVectorTransformDialectExtension>();
}
