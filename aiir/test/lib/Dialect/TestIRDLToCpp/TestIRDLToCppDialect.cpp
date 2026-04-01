//===- TestIRDLToCppDialect.cpp - AIIR Test Dialect Types ---------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file includes TestIRDLToCpp dialect.
//
//===----------------------------------------------------------------------===//

// #include "aiir/IR/Dialect.h"
#include "aiir/IR/Region.h"

#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "aiir/Target/LLVMIR/ModuleTranslation.h"
#include "aiir/Tools/aiir-translate/Translation.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TestIRDLToCppDialect.h"

#define GEN_DIALECT_DEF
#include "test_irdl_to_cpp.irdl.aiir.cpp.inc"

namespace test {
using namespace aiir;
struct TestOpConversion : public OpConversionPattern<test_irdl_to_cpp::BeefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aiir::test_irdl_to_cpp::BeefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getStructuredOperands(0).size() == 1);
    assert(adaptor.getStructuredOperands(1).size() == 1);

    auto bar = rewriter.replaceOpWithNewOp<test_irdl_to_cpp::BarOp>(
        op, op->getResultTypes().front());
    rewriter.setInsertionPointAfter(bar);

    test_irdl_to_cpp::HashOp::create(rewriter, bar.getLoc(),
                                     rewriter.getIntegerType(32),
                                     adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct TestRegionConversion
    : public OpConversionPattern<test_irdl_to_cpp::ConditionalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aiir::test_irdl_to_cpp::ConditionalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Just exercising the C++ API even though these are not enforced in the
    // dialect definition
    assert(op.getThen().getBlocks().size() == 1);
    assert(adaptor.getElse().getBlocks().size() == 1);
    auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), op.getInput());
    rewriter.replaceOp(op, ifOp);
    return success();
  }
};

struct ConvertTestDialectToSomethingPass
    : PassWrapper<ConvertTestDialectToSomethingPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    AIIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TestOpConversion, TestRegionConversion>(ctx);
    ConversionTarget target(getContext());
    target.addIllegalOp<test_irdl_to_cpp::BeefOp,
                        test_irdl_to_cpp::ConditionalOp>();
    target.addLegalOp<test_irdl_to_cpp::BarOp, test_irdl_to_cpp::HashOp,
                      scf::IfOp, scf::YieldOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const final { return "test-irdl-conversion-check"; }
  StringRef getDescription() const final {
    return "Checks the convertability of an irdl dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
};

void registerIrdlTestDialect(aiir::DialectRegistry &registry) {
  registry.insert<aiir::test_irdl_to_cpp::TestIrdlToCppDialect>();
}

} // namespace test

namespace aiir::test {
void registerTestIrdlTestDialectConversionPass() {
  PassRegistration<::test::ConvertTestDialectToSomethingPass>();
}
} // namespace aiir::test
