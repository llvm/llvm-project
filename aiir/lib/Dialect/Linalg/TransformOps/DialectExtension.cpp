//===- DialectExtension.cpp - Linalg transform dialect extension ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/Index/IR/IndexDialect.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "aiir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"

using namespace aiir;

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class LinalgTransformDialectExtension
    : public transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgTransformDialectExtension)

  using Base::Base;

  void init() {
    declareDependentDialect<linalg::LinalgDialect>();

    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<index::IndexDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<gpu::GPUDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp.inc"
        >();
    registerTransformOps<
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/TransformOps/LinalgMatchOps.cpp.inc"
        >();
  }
};
} // namespace

void aiir::linalg::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
