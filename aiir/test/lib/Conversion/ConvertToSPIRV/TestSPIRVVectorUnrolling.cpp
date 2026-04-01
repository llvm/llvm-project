//===- TestSPIRVVectorUnrolling.cpp - Test signature conversion -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "aiir/Dialect/UB/IR/UBOps.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"

namespace aiir {
namespace {

struct TestSPIRVVectorUnrolling final
    : PassWrapper<TestSPIRVVectorUnrolling, OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSPIRVVectorUnrolling)

  StringRef getArgument() const final { return "test-spirv-vector-unrolling"; }

  StringRef getDescription() const final {
    return "Test patterns that unroll vectors to types supported by SPIR-V";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<spirv::SPIRVDialect, vector::VectorDialect, ub::UBDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    (void)spirv::unrollVectorsInSignatures(op);
    (void)spirv::unrollVectorsInFuncBodies(op);
  }
};

} // namespace

namespace test {
void registerTestSPIRVVectorUnrolling() {
  PassRegistration<TestSPIRVVectorUnrolling>();
}
} // namespace test
} // namespace aiir
