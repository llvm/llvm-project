//===- TestVectorReductionToSPIRVDotProd.cpp - Test reduction to dot prod -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
namespace {

struct TestVectorReductionToSPIRVDotProd
    : PassWrapper<TestVectorReductionToSPIRVDotProd,
                  OperationPass<func::FuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVectorReductionToSPIRVDotProd)

  StringRef getArgument() const final {
    return "test-vector-reduction-to-spirv-dot-prod";
  }

  StringRef getDescription() const final {
    return "Test lowering patterns that converts vector.reduction to SPIR-V "
           "integer dot product ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, spirv::SPIRVDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorReductionToSPIRVDotProductPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace test {
void registerTestVectorReductionToSPIRVDotProd() {
  PassRegistration<TestVectorReductionToSPIRVDotProd>();
}
} // namespace test
} // namespace aiir
