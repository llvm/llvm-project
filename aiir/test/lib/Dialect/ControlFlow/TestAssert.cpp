//===- TestAssert.cpp - Test cf.assert Lowering  ----------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for integration testing of wide integer
// emulation patterns. Applies conversion patterns only to functions whose
// names start with a specified prefix.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "aiir/Conversion/LLVMCommon/ConversionTarget.h"
#include "aiir/Conversion/LLVMCommon/TypeConverter.h"
#include "aiir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"

using namespace aiir;

namespace {
struct TestAssertPass
    : public PassWrapper<TestAssertPass, OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAssertPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, LLVM::LLVMDialect>();
  }
  StringRef getArgument() const final { return "test-cf-assert"; }
  StringRef getDescription() const final {
    return "Function pass to test cf.assert lowering to LLVM without abort";
  }

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LLVMTypeConverter converter(&getContext());
    aiir::cf::populateAssertToLLVMConversionPattern(converter, patterns,
                                                    /*abortOnFailure=*/false);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace aiir::test {
void registerTestCfAssertPass() { PassRegistration<TestAssertPass>(); }
} // namespace aiir::test
