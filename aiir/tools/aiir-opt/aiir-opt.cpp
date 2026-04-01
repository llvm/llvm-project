//===- aiir-opt.cpp - AIIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for aiir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "aiir/Config/aiir-config.h"
#include "aiir/IR/AsmState.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/InitAllDialects.h"
#include "aiir/InitAllExtensions.h"
#include "aiir/InitAllPasses.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Support/FileUtilities.h"
#include "aiir/Target/LLVMIR/Dialect/All.h"
#include "aiir/Tools/aiir-opt/AiirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace aiir;

// Defined in the test directory, no public header.
namespace aiir {
void registerCloneTestPasses();
void registerConvertToTargetEnvPass();
void registerLazyLoadingTestPasses();
void registerLoopLikeInterfaceTestPasses();
void registerPassManagerTestPass();
void registerPrintSpirvAvailabilityPass();
void registerRegionTestPasses();
void registerPrintTosaAvailabilityPass();
void registerShapeFunctionTestPasses();
void registerSideEffectTestPasses();
void registerSliceAnalysisTestPass();
void registerSymbolTestPasses();
void registerTestAffineAccessAnalysisPass();
void registerTestAffineDataCopyPass();
void registerTestAffineLoopUnswitchingPass();
void registerTestAffineReifyValueBoundsPass();
void registerTestAffineWalk();
void registerTestBytecodeRoundtripPasses();
void registerTestDecomposeAffineOpPass();
void registerTestFunc();
void registerTestGpuLoweringPasses();
void registerTestGpuMemoryPromotionPass();
void registerTestLoopPermutationPass();
void registerTestMatchers();
void registerTestOperationEqualPass();
void registerTestPreserveUseListOrders();
void registerTestPrintDefUsePass();
void registerTestPrintInvalidPass();
void registerTestPrintNestingPass();
void registerTestReducer();
void registerTestSpirvEntryPointABIPass();
void registerTestSpirvModuleCombinerPass();
void registerTestTraitsPass();
void registerTosaTestQuantUtilAPIPass();
void registerVectorizerTestPass();

namespace test {
void registerCommutativityUtils();
void registerConvertCallOpPass();
void registerConvertFuncOpPass();
void registerInliner();
void registerInlinerCallback();
void registerMemRefBoundCheck();
void registerPatternsTestPass();
void registerSimpleParametricTilingPass();
void registerTestAffineLoopParametricTilingPass();
void registerTestAliasAnalysisPass();
void registerTestArithEmulateWideIntPass();
void registerTestBuiltinAttributeInterfaces();
void registerTestBuiltinDistinctAttributes();
void registerTestCallGraphPass();
void registerTestCfAssertPass();
void registerTestCFGLoopInfoPass();
void registerTestComposeSubView();
void registerTestCompositePass();
void registerTestControlFlowSink();
void registerTestConvertToSPIRVPass();
void registerTestDataLayoutQuery();
void registerTestDeadCodeAnalysisPass();
void registerTestDecomposeCallGraphTypes();
void registerTestDiagnosticsPass();
void registerTestDiagnosticsMetadataPass();
void registerTestDominancePass();
void registerTestDynamicPipelinePass();
void registerTestRemarkPass();
void registerTestEmulateNarrowTypePass();
void registerTestFooAnalysisPass();
void registerTestComposeSubView();
void registerTestMultiBuffering();
void registerTestIRVisitorsPass();
void registerTestGenericIRVisitorsPass();
void registerTestInterfaces();
void registerTestIRVisitorsPass();
void registerTestLastModifiedPass();
void registerTestLinalgDecomposeOps();
void registerTestLinalgDropUnitDims();
void registerTestLinalgElementwiseFusion();
void registerTestLinalgGreedyFusion();
void registerTestLinalgRankReduceContractionOps();
void registerTestLinalgTransforms();
void registerTestLivenessAnalysisPass();
void registerTestLivenessPass();
void registerTestLLVMLegalizePatternsPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestLoopUnrollingPass();
void registerTestLowerToArmSME();
void registerTestLowerToLLVM();
void registerTestMakeIsolatedFromAbovePass();
void registerTestMatchReductionPass();
void registerTestMathAlgebraicSimplificationPass();
void registerTestMathPolynomialApproximationPass();
void registerTestMathToVCIXPass();
void registerTestIrdlTestDialectConversionPass();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestMemRefToLLVMWithTransforms();
void registerTestReshardingPartitionPass();
void registerTestMultiBuffering();
void registerTestNextAccessPass();
void registerTestNVGPULowerings();
void registerTestOpenACC();
void registerTestOneShotModuleBufferizePass();
void registerTestOpaqueLoc();
void registerTestOpLoweringPasses();
void registerTestPadFusion();
void registerTestParallelLoopUnrollingPass();
void registerTestRecursiveTypesPass();
void registerTestSCFUpliftWhileToFor();
void registerTestSCFUtilsPass();
void registerTestSCFWhileOpBuilderPass();
void registerTestSCFWrapInZeroTripCheckPasses();
void registerTestShapeMappingPass();
void registerTestSingleFold();
void registerTestSliceAnalysisPass();
void registerTestSPIRVCPURunnerPipeline();
void registerTestSPIRVFuncSignatureConversion();
void registerTestSPIRVVectorUnrolling();
void registerTestStridedMetadataRangeAnalysisPass();
void registerTestTensorCopyInsertionPass();
void registerTestTensorLikeAndBufferLikePass();
void registerTestTensorTransforms();
void registerTestTopologicalSortAnalysisPass();
void registerTestTransformDialectEraseSchedulePass();
void registerTestPassStateExtensionCommunication();
void registerTestVectorLowerings();
void registerTestVectorReductionToSPIRVDotProd();
void registerTestVulkanRunnerPipeline();
void registerTestWrittenToPass();
void registerTestXeGPULowerings();
#if AIIR_ENABLE_PDL_IN_PATTERNMATCH
void registerTestDialectConversionPasses();
void registerTestPDLByteCodePass();
void registerTestPDLLPasses();
#endif
} // namespace test
} // namespace aiir

namespace test {
void registerTestDialect(DialectRegistry &);
void registerTestDynDialect(DialectRegistry &);
void registerTestTilingInterfaceTransformDialectExtension(DialectRegistry &);
void registerTestTransformDialectExtension(DialectRegistry &);
void registerIrdlTestDialect(DialectRegistry &);
void registerTestTransformsTransformDialectExtension(DialectRegistry &);
} // namespace test

#ifdef AIIR_INCLUDE_TESTS
static void registerTestPasses() {
  registerCloneTestPasses();
  registerConvertToTargetEnvPass();
  registerPrintTosaAvailabilityPass();
  registerLazyLoadingTestPasses();
  registerLoopLikeInterfaceTestPasses();
  registerPassManagerTestPass();
  registerPrintSpirvAvailabilityPass();
  registerRegionTestPasses();
  registerShapeFunctionTestPasses();
  registerSideEffectTestPasses();
  registerSliceAnalysisTestPass();
  registerSymbolTestPasses();
  registerTestAffineAccessAnalysisPass();
  registerTestAffineDataCopyPass();
  registerTestAffineLoopUnswitchingPass();
  registerTestAffineReifyValueBoundsPass();
  registerTestAffineWalk();
  registerTestBytecodeRoundtripPasses();
  registerTestDecomposeAffineOpPass();
  registerTestFunc();
  registerTestGpuLoweringPasses();
  registerTestGpuMemoryPromotionPass();
  registerTestLoopPermutationPass();
  registerTestMatchers();
  registerTestOperationEqualPass();
  registerTestPreserveUseListOrders();
  registerTestPrintDefUsePass();
  registerTestPrintInvalidPass();
  registerTestPrintNestingPass();
  registerTestReducer();
  registerTestSpirvEntryPointABIPass();
  registerTestSpirvModuleCombinerPass();
  registerTestTraitsPass();
  registerTosaTestQuantUtilAPIPass();
  registerVectorizerTestPass();

  aiir::test::registerCommutativityUtils();
  aiir::test::registerConvertCallOpPass();
  aiir::test::registerConvertFuncOpPass();
  aiir::test::registerInliner();
  aiir::test::registerInlinerCallback();
  aiir::test::registerMemRefBoundCheck();
  aiir::test::registerPatternsTestPass();
  aiir::test::registerSimpleParametricTilingPass();
  aiir::test::registerTestAffineLoopParametricTilingPass();
  aiir::test::registerTestAliasAnalysisPass();
  aiir::test::registerTestArithEmulateWideIntPass();
  aiir::test::registerTestBuiltinAttributeInterfaces();
  aiir::test::registerTestBuiltinDistinctAttributes();
  aiir::test::registerTestCallGraphPass();
  aiir::test::registerTestCfAssertPass();
  aiir::test::registerTestCFGLoopInfoPass();
  aiir::test::registerTestComposeSubView();
  aiir::test::registerTestCompositePass();
  aiir::test::registerTestControlFlowSink();
  aiir::test::registerTestConvertToSPIRVPass();
  aiir::test::registerTestDataLayoutQuery();
  aiir::test::registerTestDeadCodeAnalysisPass();
  aiir::test::registerTestDecomposeCallGraphTypes();
  aiir::test::registerTestDiagnosticsPass();
  aiir::test::registerTestDiagnosticsMetadataPass();
  aiir::test::registerTestDominancePass();
  aiir::test::registerTestDynamicPipelinePass();
  aiir::test::registerTestRemarkPass();
  aiir::test::registerTestEmulateNarrowTypePass();
  aiir::test::registerTestFooAnalysisPass();
  aiir::test::registerTestComposeSubView();
  aiir::test::registerTestMultiBuffering();
  aiir::test::registerTestIRVisitorsPass();
  aiir::test::registerTestGenericIRVisitorsPass();
  aiir::test::registerTestInterfaces();
  aiir::test::registerTestIrdlTestDialectConversionPass();
  aiir::test::registerTestIRVisitorsPass();
  aiir::test::registerTestLastModifiedPass();
  aiir::test::registerTestLinalgDecomposeOps();
  aiir::test::registerTestLinalgDropUnitDims();
  aiir::test::registerTestLinalgElementwiseFusion();
  aiir::test::registerTestLinalgGreedyFusion();
  aiir::test::registerTestLinalgRankReduceContractionOps();
  aiir::test::registerTestLinalgTransforms();
  aiir::test::registerTestLivenessAnalysisPass();
  aiir::test::registerTestLivenessPass();
  aiir::test::registerTestLLVMLegalizePatternsPass();
  aiir::test::registerTestLoopFusion();
  aiir::test::registerTestLoopMappingPass();
  aiir::test::registerTestLoopUnrollingPass();
  aiir::test::registerTestLowerToArmSME();
  aiir::test::registerTestLowerToLLVM();
  aiir::test::registerTestMakeIsolatedFromAbovePass();
  aiir::test::registerTestMatchReductionPass();
  aiir::test::registerTestMathAlgebraicSimplificationPass();
  aiir::test::registerTestMathPolynomialApproximationPass();
  aiir::test::registerTestMathToVCIXPass();
  aiir::test::registerTestMemRefDependenceCheck();
  aiir::test::registerTestMemRefStrideCalculation();
  aiir::test::registerTestMemRefToLLVMWithTransforms();
  aiir::test::registerTestReshardingPartitionPass();
  aiir::test::registerTestMultiBuffering();
  aiir::test::registerTestNextAccessPass();
  aiir::test::registerTestNVGPULowerings();
  aiir::test::registerTestOpenACC();
  aiir::test::registerTestOneShotModuleBufferizePass();
  aiir::test::registerTestOpaqueLoc();
  aiir::test::registerTestOpLoweringPasses();
  aiir::test::registerTestPadFusion();
  aiir::test::registerTestParallelLoopUnrollingPass();
  aiir::test::registerTestRecursiveTypesPass();
  aiir::test::registerTestSCFUpliftWhileToFor();
  aiir::test::registerTestSCFUtilsPass();
  aiir::test::registerTestSCFWhileOpBuilderPass();
  aiir::test::registerTestSCFWrapInZeroTripCheckPasses();
  aiir::test::registerTestShapeMappingPass();
  aiir::test::registerTestSingleFold();
  aiir::test::registerTestSliceAnalysisPass();
  aiir::test::registerTestSPIRVCPURunnerPipeline();
  aiir::test::registerTestSPIRVFuncSignatureConversion();
  aiir::test::registerTestSPIRVVectorUnrolling();
  aiir::test::registerTestStridedMetadataRangeAnalysisPass();
  aiir::test::registerTestTensorCopyInsertionPass();
  aiir::test::registerTestTensorLikeAndBufferLikePass();
  aiir::test::registerTestTensorTransforms();
  aiir::test::registerTestTopologicalSortAnalysisPass();
  aiir::test::registerTestTransformDialectEraseSchedulePass();
  aiir::test::registerTestPassStateExtensionCommunication();
  aiir::test::registerTestVectorLowerings();
  aiir::test::registerTestVectorReductionToSPIRVDotProd();
  aiir::test::registerTestVulkanRunnerPipeline();
  aiir::test::registerTestWrittenToPass();
  aiir::test::registerTestXeGPULowerings();
#if AIIR_ENABLE_PDL_IN_PATTERNMATCH
  aiir::test::registerTestDialectConversionPasses();
  aiir::test::registerTestPDLByteCodePass();
  aiir::test::registerTestPDLLPasses();
#endif
}
#endif

int main(int argc, char **argv) {
  registerAllPasses();
#ifdef AIIR_INCLUDE_TESTS
  registerTestPasses();
#endif
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);

  // TODO: Remove this and the corresponding AIIRToLLVMIRTranslationRegistration
  // cmake dependency when a safe dialect interface registration mechanism is
  // implemented, see D157703 (and corresponding note on the declaration).
  registerAllGPUToLLVMIRTranslations(registry);

#ifdef AIIR_INCLUDE_TESTS
  ::test::registerIrdlTestDialect(registry);
  ::test::registerTestDialect(registry);
  ::test::registerTestDynDialect(registry);
  ::test::registerTestTilingInterfaceTransformDialectExtension(registry);
  ::test::registerTestTransformDialectExtension(registry);
  ::test::registerTestTransformsTransformDialectExtension(registry);
#endif
  return aiir::asMainReturnCode(aiir::AiirOptMain(
      argc, argv, "AIIR modular optimizer driver\n", registry));
}
