//===- HLFIRExtensionPointsTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the HLFIR extension points of the HLFIR-to-FIR pass pipeline.
//
// The callbacks run when the pipeline is built, so no IR is needed: building
// the pipeline is enough to observe them.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "flang/Optimizer/Passes/Pipelines.h"
#include "flang/Tools/CrossToolHelpers.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace {

struct MarkerPass : public mlir::PassWrapper<MarkerPass,
                        mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkerPass)

  llvm::StringRef getArgument() const override { return "ep-marker"; }
  llvm::StringRef getDescription() const override {
    return "No-op pass used to locate an extension point in a pipeline";
  }
  void runOnOperation() override {}
};

TEST(HLFIRExtensionPoint, CallbacksAreInvokedInOrder) {
  mlir::MLIRContext context;
  mlir::PassManager pm(&context, mlir::ModuleOp::getOperationName());
  MLIRToLLVMPassPipelineConfig config(llvm::OptimizationLevel::O2);

  std::vector<std::string> order;
  size_t earlySizeAtCall = ~size_t{0};

  config.registerHLFIROptEarlyEPCallbacks(
      [&](mlir::PassManager &nestedPm, llvm::OptimizationLevel) {
        order.push_back("early");
        earlySizeAtCall = nestedPm.size();
      });
  config.registerHLFIROptLastEPCallbacks(
      [&](mlir::PassManager &, llvm::OptimizationLevel) {
        order.push_back("last");
      });

  fir::createHLFIRToFIRPassPipeline(pm, fir::EnableOpenMP::None, config);

  ASSERT_EQ(order.size(), 2u);
  EXPECT_EQ(order[0], "early");
  EXPECT_EQ(order[1], "last");
  EXPECT_EQ(earlySizeAtCall, 0u);
  EXPECT_GT(pm.size(), 0u);
}

TEST(HLFIRExtensionPoint, CallbacksAreInvokedAtEveryOptLevel) {
  for (llvm::OptimizationLevel level :
      {llvm::OptimizationLevel::O0, llvm::OptimizationLevel::O1,
          llvm::OptimizationLevel::O2, llvm::OptimizationLevel::O3}) {
    mlir::MLIRContext context;
    mlir::PassManager pm(&context, mlir::ModuleOp::getOperationName());
    MLIRToLLVMPassPipelineConfig config(level);

    int earlyCount = 0;
    int lastCount = 0;
    llvm::OptimizationLevel seenLevel = llvm::OptimizationLevel::O0;
    config.registerHLFIROptEarlyEPCallbacks(
        [&](mlir::PassManager &, llvm::OptimizationLevel cbLevel) {
          ++earlyCount;
          seenLevel = cbLevel;
        });
    config.registerHLFIROptLastEPCallbacks(
        [&](mlir::PassManager &, llvm::OptimizationLevel) { ++lastCount; });

    fir::createHLFIRToFIRPassPipeline(pm, fir::EnableOpenMP::None, config);

    EXPECT_EQ(earlyCount, 1);
    EXPECT_EQ(lastCount, 1);
    EXPECT_EQ(seenLevel, level);
  }
}

TEST(HLFIRExtensionPoint, MarkersAreAtTheDocumentedPositions) {
  mlir::MLIRContext context;
  mlir::PassManager pm(&context, mlir::ModuleOp::getOperationName());
  MLIRToLLVMPassPipelineConfig config(llvm::OptimizationLevel::O2);

  config.registerHLFIROptEarlyEPCallbacks(
      [](mlir::PassManager &nestedPm, llvm::OptimizationLevel) {
        nestedPm.addPass(std::make_unique<MarkerPass>());
      });
  config.registerHLFIROptLastEPCallbacks(
      [](mlir::PassManager &nestedPm, llvm::OptimizationLevel) {
        nestedPm.addPass(std::make_unique<MarkerPass>());
      });

  fir::createHLFIRToFIRPassPipeline(pm, fir::EnableOpenMP::None, config);

  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  pm.printAsTextualPipeline(os);

  size_t earlyMarker = pipeline.find("ep-marker");
  ASSERT_NE(earlyMarker, std::string::npos) << pipeline;
  size_t lastMarker = pipeline.find("ep-marker", earlyMarker + 1);
  ASSERT_NE(lastMarker, std::string::npos) << pipeline;

  size_t simplify = pipeline.find("simplify-hlfir-intrinsics");
  ASSERT_NE(simplify, std::string::npos) << pipeline;
  size_t lowerIntrinsics = pipeline.find("lower-hlfir-intrinsics");
  ASSERT_NE(lowerIntrinsics, std::string::npos) << pipeline;

  EXPECT_LT(earlyMarker, simplify) << pipeline;
  EXPECT_GT(lastMarker, simplify) << pipeline;
  EXPECT_LT(lastMarker, lowerIntrinsics) << pipeline;
}

} // namespace
