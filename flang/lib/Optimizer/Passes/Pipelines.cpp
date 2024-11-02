//===-- Pipelines.cpp -- FIR pass pipelines ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// This file defines some utilties to setup FIR pass pipelines. These are
/// common to flang and the test tools.

#include "flang/Optimizer/Passes/Pipelines.h"

namespace fir {

void addNestedPassToAllTopLevelOperations(mlir::PassManager &pm,
                                          PassConstructor ctor) {
  addNestedPassToOps<mlir::func::FuncOp, mlir::omp::DeclareReductionOp,
                     mlir::omp::PrivateClauseOp, fir::GlobalOp>(pm, ctor);
}

void addNestedPassToAllTopLevelOperationsConditionally(
    mlir::PassManager &pm, llvm::cl::opt<bool> &disabled,
    PassConstructor ctor) {
  if (!disabled)
    addNestedPassToAllTopLevelOperations(pm, ctor);
}

void addCanonicalizerPassWithoutRegionSimplification(mlir::OpPassManager &pm) {
  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  pm.addPass(mlir::createCanonicalizerPass(config));
}

void addCfgConversionPass(mlir::PassManager &pm,
                          const MLIRToLLVMPassPipelineConfig &config) {
  if (config.NSWOnLoopVarInc)
    addNestedPassToAllTopLevelOperationsConditionally(
        pm, disableCfgConversion, fir::createCFGConversionPassWithNSW);
  else
    addNestedPassToAllTopLevelOperationsConditionally(pm, disableCfgConversion,
                                                      fir::createCFGConversion);
}

void addAVC(mlir::PassManager &pm, const llvm::OptimizationLevel &optLevel) {
  ArrayValueCopyOptions options;
  options.optimizeConflicts = optLevel.isOptimizingForSpeed();
  addNestedPassConditionally<mlir::func::FuncOp>(
      pm, disableFirAvc, [&]() { return createArrayValueCopyPass(options); });
}

void addMemoryAllocationOpt(mlir::PassManager &pm) {
  addNestedPassConditionally<mlir::func::FuncOp>(pm, disableFirMao, [&]() {
    return fir::createMemoryAllocationOpt(
        {dynamicArrayStackToHeapAllocation, arrayStackAllocationThreshold});
  });
}

void addCodeGenRewritePass(mlir::PassManager &pm, bool preserveDeclare) {
  fir::CodeGenRewriteOptions options;
  options.preserveDeclare = preserveDeclare;
  addPassConditionally(pm, disableCodeGenRewrite,
                       [&]() { return fir::createCodeGenRewrite(options); });
}

void addTargetRewritePass(mlir::PassManager &pm) {
  addPassConditionally(pm, disableTargetRewrite,
                       []() { return fir::createTargetRewritePass(); });
}

mlir::LLVM::DIEmissionKind
getEmissionKind(llvm::codegenoptions::DebugInfoKind kind) {
  switch (kind) {
  case llvm::codegenoptions::DebugInfoKind::FullDebugInfo:
    return mlir::LLVM::DIEmissionKind::Full;
  case llvm::codegenoptions::DebugInfoKind::DebugLineTablesOnly:
    return mlir::LLVM::DIEmissionKind::LineTablesOnly;
  default:
    return mlir::LLVM::DIEmissionKind::None;
  }
}

void addDebugInfoPass(mlir::PassManager &pm,
                      llvm::codegenoptions::DebugInfoKind debugLevel,
                      llvm::OptimizationLevel optLevel,
                      llvm::StringRef inputFilename) {
  fir::AddDebugInfoOptions options;
  options.debugLevel = getEmissionKind(debugLevel);
  options.isOptimized = optLevel != llvm::OptimizationLevel::O0;
  options.inputFilename = inputFilename;
  addPassConditionally(pm, disableDebugInfo,
                       [&]() { return fir::createAddDebugInfoPass(options); });
}

void addFIRToLLVMPass(mlir::PassManager &pm,
                      const MLIRToLLVMPassPipelineConfig &config) {
  fir::FIRToLLVMPassOptions options;
  options.ignoreMissingTypeDescriptors = ignoreMissingTypeDescriptors;
  options.applyTBAA = config.AliasAnalysis;
  options.forceUnifiedTBAATree = useOldAliasTags;
  options.typeDescriptorsRenamedForAssembly =
      !disableCompilerGeneratedNamesConversion;
  addPassConditionally(pm, disableFirToLlvmIr,
                       [&]() { return fir::createFIRToLLVMPass(options); });
  // The dialect conversion framework may leave dead unrealized_conversion_cast
  // ops behind, so run reconcile-unrealized-casts to clean them up.
  addPassConditionally(pm, disableFirToLlvmIr, [&]() {
    return mlir::createReconcileUnrealizedCastsPass();
  });
}

void addLLVMDialectToLLVMPass(mlir::PassManager &pm,
                              llvm::raw_ostream &output) {
  addPassConditionally(pm, disableLlvmIrToLlvm, [&]() {
    return fir::createLLVMDialectToLLVMPass(output);
  });
}

void addBoxedProcedurePass(mlir::PassManager &pm) {
  addPassConditionally(pm, disableBoxedProcedureRewrite,
                       [&]() { return fir::createBoxedProcedurePass(); });
}

void addExternalNameConversionPass(mlir::PassManager &pm,
                                   bool appendUnderscore) {
  addPassConditionally(pm, disableExternalNameConversion, [&]() {
    return fir::createExternalNameConversion({appendUnderscore});
  });
}

void addCompilerGeneratedNamesConversionPass(mlir::PassManager &pm) {
  addPassConditionally(pm, disableCompilerGeneratedNamesConversion, [&]() {
    return fir::createCompilerGeneratedNamesConversion();
  });
}

// Use inliner extension point callback to register the default inliner pass.
void registerDefaultInlinerPass(MLIRToLLVMPassPipelineConfig &config) {
  config.registerFIRInlinerCallback(
      [](mlir::PassManager &pm, llvm::OptimizationLevel level) {
        llvm::StringMap<mlir::OpPassManager> pipelines;
        // The default inliner pass adds the canonicalizer pass with the default
        // configuration.
        pm.addPass(mlir::createInlinerPass(
            pipelines, addCanonicalizerPassWithoutRegionSimplification));
      });
}

/// Create a pass pipeline for running default optimization passes for
/// incremental conversion of FIR.
///
/// \param pm - MLIR pass manager that will hold the pipeline definition
void createDefaultFIROptimizerPassPipeline(mlir::PassManager &pm,
                                           MLIRToLLVMPassPipelineConfig &pc) {
  // Early Optimizer EP Callback
  pc.invokeFIROptEarlyEPCallbacks(pm, pc.OptLevel);

  // simplify the IR
  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  pm.addPass(mlir::createCSEPass());
  fir::addAVC(pm, pc.OptLevel);
  addNestedPassToAllTopLevelOperations(pm, fir::createCharacterConversion);
  pm.addPass(mlir::createCanonicalizerPass(config));
  pm.addPass(fir::createSimplifyRegionLite());
  if (pc.OptLevel.isOptimizingForSpeed()) {
    // These passes may increase code size.
    pm.addPass(fir::createSimplifyIntrinsics());
    pm.addPass(fir::createAlgebraicSimplificationPass(config));
    if (enableConstantArgumentGlobalisation)
      pm.addPass(fir::createConstantArgumentGlobalisationOpt());
  }

  if (pc.LoopVersioning)
    pm.addPass(fir::createLoopVersioning());

  pm.addPass(mlir::createCSEPass());

  if (pc.StackArrays)
    pm.addPass(fir::createStackArrays());
  else
    fir::addMemoryAllocationOpt(pm);

  // FIR Inliner Callback
  pc.invokeFIRInlinerCallback(pm, pc.OptLevel);

  pm.addPass(fir::createSimplifyRegionLite());
  pm.addPass(mlir::createCSEPass());

  // Polymorphic types
  pm.addPass(fir::createPolymorphicOpConversion());
  pm.addPass(fir::createAssumedRankOpConversion());

  if (pc.AliasAnalysis && !disableFirAliasTags && !useOldAliasTags)
    pm.addPass(fir::createAddAliasTags());

  addNestedPassToAllTopLevelOperations(pm, fir::createStackReclaim);
  // convert control flow to CFG form
  fir::addCfgConversionPass(pm, pc);
  pm.addPass(mlir::createConvertSCFToCFPass());

  pm.addPass(mlir::createCanonicalizerPass(config));
  pm.addPass(fir::createSimplifyRegionLite());
  pm.addPass(mlir::createCSEPass());

  // Last Optimizer EP Callback
  pc.invokeFIROptLastEPCallbacks(pm, pc.OptLevel);
}

/// Create a pass pipeline for lowering from HLFIR to FIR
///
/// \param pm - MLIR pass manager that will hold the pipeline definition
/// \param optLevel - optimization level used for creating FIR optimization
///   passes pipeline
void createHLFIRToFIRPassPipeline(mlir::PassManager &pm,
                                  llvm::OptimizationLevel optLevel) {
  if (optLevel.isOptimizingForSpeed()) {
    addCanonicalizerPassWithoutRegionSimplification(pm);
    addNestedPassToAllTopLevelOperations(pm,
                                         hlfir::createSimplifyHLFIRIntrinsics);
  }
  addNestedPassToAllTopLevelOperations(pm, hlfir::createInlineElementals);
  if (optLevel.isOptimizingForSpeed()) {
    addCanonicalizerPassWithoutRegionSimplification(pm);
    pm.addPass(mlir::createCSEPass());
    addNestedPassToAllTopLevelOperations(pm,
                                         hlfir::createOptimizedBufferization);
  }
  pm.addPass(hlfir::createLowerHLFIROrderedAssignments());
  pm.addPass(hlfir::createLowerHLFIRIntrinsics());
  pm.addPass(hlfir::createBufferizeHLFIR());
  pm.addPass(hlfir::createConvertHLFIRtoFIR());
}

/// Create a pass pipeline for handling certain OpenMP transformations needed
/// prior to FIR lowering.
///
/// WARNING: These passes must be run immediately after the lowering to ensure
/// that the FIR is correct with respect to OpenMP operations/attributes.
///
/// \param pm - MLIR pass manager that will hold the pipeline definition.
/// \param isTargetDevice - Whether code is being generated for a target device
/// rather than the host device.
void createOpenMPFIRPassPipeline(mlir::PassManager &pm, bool isTargetDevice) {
  pm.addPass(flangomp::createMapInfoFinalizationPass());
  pm.addPass(flangomp::createMarkDeclareTargetPass());
  if (isTargetDevice)
    pm.addPass(flangomp::createFunctionFilteringPass());
}

void createDebugPasses(mlir::PassManager &pm,
                       llvm::codegenoptions::DebugInfoKind debugLevel,
                       llvm::OptimizationLevel OptLevel,
                       llvm::StringRef inputFilename) {
  if (debugLevel != llvm::codegenoptions::NoDebugInfo)
    addDebugInfoPass(pm, debugLevel, OptLevel, inputFilename);
}

void createDefaultFIRCodeGenPassPipeline(mlir::PassManager &pm,
                                         MLIRToLLVMPassPipelineConfig config,
                                         llvm::StringRef inputFilename) {
  fir::addBoxedProcedurePass(pm);
  addNestedPassToAllTopLevelOperations(pm, fir::createAbstractResultOpt);
  fir::addCodeGenRewritePass(
      pm, (config.DebugInfo != llvm::codegenoptions::NoDebugInfo));
  fir::addTargetRewritePass(pm);
  fir::addCompilerGeneratedNamesConversionPass(pm);
  fir::addExternalNameConversionPass(pm, config.Underscoring);
  fir::createDebugPasses(pm, config.DebugInfo, config.OptLevel, inputFilename);

  if (config.VScaleMin != 0)
    pm.addPass(fir::createVScaleAttr({{config.VScaleMin, config.VScaleMax}}));

  // Add function attributes
  mlir::LLVM::framePointerKind::FramePointerKind framePointerKind;

  if (config.FramePointerKind != llvm::FramePointerKind::None ||
      config.NoInfsFPMath || config.NoNaNsFPMath || config.ApproxFuncFPMath ||
      config.NoSignedZerosFPMath || config.UnsafeFPMath) {
    if (config.FramePointerKind == llvm::FramePointerKind::NonLeaf)
      framePointerKind =
          mlir::LLVM::framePointerKind::FramePointerKind::NonLeaf;
    else if (config.FramePointerKind == llvm::FramePointerKind::All)
      framePointerKind = mlir::LLVM::framePointerKind::FramePointerKind::All;
    else
      framePointerKind = mlir::LLVM::framePointerKind::FramePointerKind::None;

    pm.addPass(fir::createFunctionAttr(
        {framePointerKind, config.NoInfsFPMath, config.NoNaNsFPMath,
         config.ApproxFuncFPMath, config.NoSignedZerosFPMath,
         config.UnsafeFPMath}));
  }

  fir::addFIRToLLVMPass(pm, config);
}

/// Create a pass pipeline for lowering from MLIR to LLVM IR
///
/// \param pm - MLIR pass manager that will hold the pipeline definition
/// \param optLevel - optimization level used for creating FIR optimization
///   passes pipeline
void createMLIRToLLVMPassPipeline(mlir::PassManager &pm,
                                  MLIRToLLVMPassPipelineConfig &config,
                                  llvm::StringRef inputFilename) {
  fir::createHLFIRToFIRPassPipeline(pm, config.OptLevel);

  // Add default optimizer pass pipeline.
  fir::createDefaultFIROptimizerPassPipeline(pm, config);

  // Add codegen pass pipeline.
  fir::createDefaultFIRCodeGenPassPipeline(pm, config, inputFilename);
}

} // namespace fir
