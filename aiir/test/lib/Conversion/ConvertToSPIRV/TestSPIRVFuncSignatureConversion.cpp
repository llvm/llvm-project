//===- TestSPIRVFuncSignatureConversion.cpp - Test signature conversion -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
namespace {

struct TestSPIRVFuncSignatureConversion final
    : PassWrapper<TestSPIRVFuncSignatureConversion, OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSPIRVFuncSignatureConversion)

  StringRef getArgument() const final {
    return "test-spirv-func-signature-conversion";
  }

  StringRef getDescription() const final {
    return "Test patterns that convert vector inputs and results in function "
           "signatures";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, spirv::SPIRVDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    (void)spirv::unrollVectorsInSignatures(op);
  }
};

} // namespace

namespace test {
void registerTestSPIRVFuncSignatureConversion() {
  PassRegistration<TestSPIRVFuncSignatureConversion>();
}
} // namespace test
} // namespace aiir
