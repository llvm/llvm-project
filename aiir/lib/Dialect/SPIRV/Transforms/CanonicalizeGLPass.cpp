//===- CanonicalizeGLPass.cpp - GLSL Related Canonicalization Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/SPIRV/Transforms/Passes.h"

#include "aiir/Dialect/SPIRV/IR/SPIRVGLCanonicalization.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVCANONICALIZEGLPASS
#include "aiir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace aiir

using namespace aiir;

namespace {
class CanonicalizeGLPass final
    : public spirv::impl::SPIRVCanonicalizeGLPassBase<CanonicalizeGLPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    spirv::populateSPIRVGLCanonicalizationPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
