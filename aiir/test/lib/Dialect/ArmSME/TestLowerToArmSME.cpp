//===- TestLowerToArmSME.cpp - Test lowering to ArmSME as a sink pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing the lowering to ArmSME as a
// generally usable sink pass.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "aiir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "aiir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "aiir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "aiir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "aiir/Conversion/VectorToSCF/VectorToSCF.h"
#include "aiir/Dialect/ArmSME/Transforms/Passes.h"
#include "aiir/Dialect/ArmSVE/Transforms/Passes.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/DialectRegistry.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Pass/PassOptions.h"
#include "aiir/Transforms/Passes.h"

using namespace aiir;

namespace {
struct TestLowerToArmSMEOptions
    : public PassPipelineOptions<TestLowerToArmSMEOptions> {
  PassOptions::Option<bool> fuseOuterProducts{
      *this, "fuse-outer-products",
      llvm::cl::desc("Fuse outer product operations via "
                     "'-arm-sme-outer-product-fusion' pass"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> dumpTileLiveRanges{
      *this, "dump-tile-live-ranges",
      llvm::cl::desc("Dump the live ranges of SME tiles (for debugging)"),
      llvm::cl::init(false)};
};

void buildTestLowerToArmSME(OpPassManager &pm,
                            const TestLowerToArmSMEOptions &options) {
  // Legalize vector operations so they can be converted to ArmSME.
  pm.addPass(arm_sme::createVectorLegalizationPass());

  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Passes that convert operations on vectors to ArmSME operations.

  // Convert Arith to ArmSME.
  pm.addPass(createArithToArmSMEConversionPass());
  // Convert Vector to ArmSME.
  pm.addPass(createConvertVectorToArmSMEPass());

  // Fuse outer products.
  if (options.fuseOuterProducts)
    pm.addPass(arm_sme::createOuterProductFusionPass());

  // Convert operations on high-level vectors to loops.

  // Convert ArmSME to SCF.
  pm.addPass(createConvertArmSMEToSCFPass());

  // Convert Vector to SCF (with full unroll enabled).
  pm.addPass(createConvertVectorToSCFPass(
      VectorTransferToSCFOptions().enableFullUnroll()));

  // Enable streaming-mode and ZA.
  pm.addPass(arm_sme::createEnableArmStreamingPass(
      arm_sme::ArmStreamingMode::StreamingLocally, arm_sme::ArmZaMode::NewZA,
      /*ifRequiredByOps=*/true));

  // Convert SCF to CF (required for ArmSME tile allocation).
  pm.addPass(createSCFToControlFlowPass());

  // Convert ArmSME to LLVM.
  pm.addNestedPass<func::FuncOp>(
      createConvertArmSMEToLLVMPass(options.dumpTileLiveRanges));

  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}
} // namespace

namespace aiir {
namespace test {
void registerTestLowerToArmSME() {
  PassPipelineRegistration<TestLowerToArmSMEOptions>(
      "test-lower-to-arm-sme",
      "An example pipeline to lower operations on vectors (arith, vector) to "
      "LLVM via ArmSME.",
      buildTestLowerToArmSME);
}
} // namespace test
} // namespace aiir
