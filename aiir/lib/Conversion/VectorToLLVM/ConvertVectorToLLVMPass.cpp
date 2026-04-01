//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"

#include "aiir/Conversion/LLVMCommon/ConversionTarget.h"
#include "aiir/Conversion/LLVMCommon/TypeConverter.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "aiir/Dialect/ArmNeon/Transforms.h"
#include "aiir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "aiir/Dialect/ArmSVE/Transforms/Transforms.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "aiir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "aiir/Dialect/X86/Transforms.h"
#include "aiir/Dialect/X86/X86Dialect.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTVECTORTOLLVMPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using namespace aiir::vector;

namespace {
struct ConvertVectorToLLVMPass
    : public impl::ConvertVectorToLLVMPassBase<ConvertVectorToLLVMPass> {

  using Base::Base;

  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<tensor::TensorDialect>();
    if (armNeon)
      registry.insert<arm_neon::ArmNeonDialect>();
    if (armSVE)
      registry.insert<arm_sve::ArmSVEDialect>();
    if (x86)
      registry.insert<x86::X86Dialect>();
  }
  void runOnOperation() override;
};
} // namespace

void ConvertVectorToLLVMPass::runOnOperation() {
  // Perform progressive lowering of operations on slices and all contraction
  // operations. Also materializes masks, lowers vector.step, rank-reduces FMA,
  // applies folding and DCE.
  {
    RewritePatternSet patterns(&getContext());
    populateVectorToVectorCanonicalizationPatterns(patterns);
    populateVectorBitCastLoweringPatterns(patterns);
    populateVectorBroadcastLoweringPatterns(patterns);
    populateVectorContractLoweringPatterns(patterns, vectorContractLowering);
    if (vectorContractLowering == vector::VectorContractLowering::LLVMIntr) {
      // This pattern creates a dependency on the LLVM dialect, hence we don't
      // include it in `populateVectorContractLoweringPatterns` that is part of
      // the Vector dialect (and should not depend on LLVM).
      populateVectorContractToMatrixMultiply(patterns);
    }
    populateVectorMaskOpLoweringPatterns(patterns);
    populateVectorShapeCastLoweringPatterns(patterns);
    populateVectorInterleaveLoweringPatterns(patterns);
    populateVectorTransposeLoweringPatterns(patterns, vectorTransposeLowering);
    if (vectorTransposeLowering == vector::VectorTransposeLowering::LLVMIntr) {
      // This pattern creates a dependency on the LLVM dialect, hence we don't
      // include it in `populateVectorTransposeLoweringPatterns` that is part of
      // the Vector dialect (and should not depend on LLVM).
      populateVectorTransposeToFlatTranspose(patterns);
    }
    // Vector transfer ops with rank > 1 should be lowered with VectorToSCF.
    populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
    populateVectorMaskMaterializationPatterns(patterns,
                                              force32BitVectorIndices);
    populateVectorInsertExtractStridedSliceTransforms(patterns);
    populateVectorStepLoweringPatterns(patterns);
    populateVectorRankReducingFMAPattern(patterns);
    populateVectorGatherLoweringPatterns(patterns);
    populateVectorFromElementsUnrollPatterns(patterns);
    populateVectorToElementsUnrollPatterns(patterns);
    if (armI8MM) {
      if (armNeon)
        arm_neon::populateLowerContractionToNeonI8MMPatterns(patterns);
      if (armSVE)
        populateLowerContractionToSVEI8MMPatterns(patterns);
    }
    if (armBF16) {
      if (armNeon)
        arm_neon::populateLowerContractionToNeonBFMMLAPatterns(patterns);
      if (armSVE)
        populateLowerContractionToSVEBFMMLAPatterns(patterns);
    }
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }

  // Convert to the LLVM IR dialect.
  LowerToLLVMOptions options(&getContext());
  LLVMTypeConverter converter(&getContext(), options);
  RewritePatternSet patterns(&getContext());
  populateVectorTransferLoweringPatterns(patterns);
  populateVectorToLLVMConversionPatterns(
      converter, patterns, reassociateFPReductions, force32BitVectorIndices,
      useVectorAlignment);

  // Architecture specific augmentations.
  LLVMConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  if (armNeon) {
    // TODO: we may or may not want to include in-dialect lowering to
    // LLVM-compatible operations here. So far, all operations in the dialect
    // can be translated to LLVM IR so there is no conversion necessary.
    target.addLegalDialect<arm_neon::ArmNeonDialect>();
  }
  if (armSVE) {
    configureArmSVELegalizeForExportTarget(target);
    populateArmSVELegalizeForLLVMExportPatterns(converter, patterns);
  }
  if (x86) {
    configureX86LegalizeForExportTarget(target);
    populateX86LegalizeForLLVMExportPatterns(converter, patterns);
  }

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
