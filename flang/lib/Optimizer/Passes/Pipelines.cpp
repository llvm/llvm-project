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
#include "llvm/Support/CommandLine.h"

/// Force setting the no-alias attribute on fuction arguments when possible.
static llvm::cl::opt<bool> forceNoAlias("force-no-alias", llvm::cl::Hidden,
                                        llvm::cl::init(true));

namespace fir {

template <typename F>
void addNestedPassToAllTopLevelOperations(mlir::PassManager &pm, F ctor) {
  addNestedPassToOps<F, mlir::func::FuncOp, mlir::omp::DeclareReductionOp,
                     mlir::omp::PrivateClauseOp, fir::GlobalOp>(pm, ctor);
}

template <typename F>
void addPassToGPUModuleOperations(mlir::PassManager &pm, F ctor) {
  mlir::OpPassManager &nestPM = pm.nest<mlir::gpu::GPUModuleOp>();
  nestPM.addNestedPass<mlir::func::FuncOp>(ctor());
  nestPM.addNestedPass<mlir::gpu::GPUFuncOp>(ctor());
}

template <typename F>
void addNestedPassToAllTopLevelOperationsConditionally(
    mlir::PassManager &pm, llvm::cl::opt<bool> &disabled, F ctor) {
  if (!disabled)
    addNestedPassToAllTopLevelOperations<F>(pm, ctor);
}

void addCanonicalizerPassWithoutRegionSimplification(mlir::OpPassManager &pm) {
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  pm.addPass(mlir::createCanonicalizerPass(config));
}

void addCfgConversionPass(mlir::PassManager &pm,
                          const MLIRToLLVMPassPipelineConfig &config) {
  fir::CFGConversionOptions options;
  if (!config.NSWOnLoopVarInc)
    options.setNSW = false;
  addNestedPassToAllTopLevelOperationsConditionally(
      pm, disableCfgConversion, [&]() { return createCFGConversion(options); });
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
  options.skipExternalRttiDefinition = skipExternalRttiDefinition;
  options.applyTBAA = config.AliasAnalysis;
  options.forceUnifiedTBAATree = useOldAliasTags;
  options.typeDescriptorsRenamedForAssembly =
      !disableCompilerGeneratedNamesConversion;
  options.ComplexRange = config.ComplexRange;
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
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  pm.addPass(mlir::createCSEPass());
  fir::addAVC(pm, pc.OptLevel);
  addNestedPassToAllTopLevelOperations<PassConstructor>(
      pm, fir::createCharacterConversion);
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

  // Optimize redundant array repacking operations,
  // if the source is known to be contiguous.
  if (pc.OptLevel.isOptimizingForSpeed())
    pm.addPass(fir::createOptimizeArrayRepacking());
  pm.addPass(fir::createLowerRepackArraysPass());
  // Expand FIR operations that may use SCF dialect for their
  // implementation. This is a mandatory pass.
  pm.addPass(fir::createSimplifyFIROperations(
      {/*preferInlineImplementation=*/pc.OptLevel.isOptimizingForSpeed()}));

  if (pc.AliasAnalysis && !disableFirAliasTags && !useOldAliasTags)
    pm.addPass(fir::createAddAliasTags());

  addNestedPassToAllTopLevelOperations<PassConstructor>(
      pm, fir::createStackReclaim);
  // convert control flow to CFG form
  fir::addCfgConversionPass(pm, pc);
  pm.addPass(mlir::createSCFToControlFlowPass());

  pm.addPass(mlir::createCanonicalizerPass(config));
  pm.addPass(fir::createSimplifyRegionLite());
  pm.addPass(mlir::createCSEPass());

  if (pc.OptLevel.isOptimizingForSpeed())
    pm.addPass(fir::createSetRuntimeCallAttributes());

  // Last Optimizer EP Callback
  pc.invokeFIROptLastEPCallbacks(pm, pc.OptLevel);
}

/// Create a pass pipeline for lowering from HLFIR to FIR
///
/// \param pm - MLIR pass manager that will hold the pipeline definition
/// \param optLevel - optimization level used for creating FIR optimization
///   passes pipeline
void createHLFIRToFIRPassPipeline(mlir::PassManager &pm,
                                  EnableOpenMP enableOpenMP,
                                  llvm::OptimizationLevel optLevel) {
  if (optLevel.isOptimizingForSpeed()) {
    addCanonicalizerPassWithoutRegionSimplification(pm);
    addNestedPassToAllTopLevelOperations<PassConstructor>(
        pm, hlfir::createSimplifyHLFIRIntrinsics);
  }
  addNestedPassToAllTopLevelOperations<PassConstructor>(
      pm, hlfir::createInlineElementals);
  if (optLevel.isOptimizingForSpeed()) {
    addCanonicalizerPassWithoutRegionSimplification(pm);
    pm.addPass(mlir::createCSEPass());
    // Run SimplifyHLFIRIntrinsics pass late after CSE,
    // and allow introducing operations with new side effects.
    addNestedPassToAllTopLevelOperations<PassConstructor>(pm, []() {
      return hlfir::createSimplifyHLFIRIntrinsics(
          {/*allowNewSideEffects=*/true});
    });
    addNestedPassToAllTopLevelOperations<PassConstructor>(
        pm, hlfir::createPropagateFortranVariableAttributes);
    addNestedPassToAllTopLevelOperations<PassConstructor>(
        pm, hlfir::createOptimizedBufferization);
    addNestedPassToAllTopLevelOperations<PassConstructor>(
        pm, hlfir::createInlineHLFIRAssign);

    if (optLevel == llvm::OptimizationLevel::O3) {
      addNestedPassToAllTopLevelOperations<PassConstructor>(
          pm, hlfir::createInlineHLFIRCopyIn);
    }
  }
  pm.addPass(hlfir::createLowerHLFIROrderedAssignments());
  pm.addPass(hlfir::createLowerHLFIRIntrinsics());

  hlfir::BufferizeHLFIROptions bufferizeOptions;
  // For opt-for-speed, avoid running any of the loops resulting
  // from hlfir.elemental lowering, if the result is an empty array.
  // This helps to avoid long running loops for elementals with
  // shapes like (0, HUGE).
  if (optLevel.isOptimizingForSpeed())
    bufferizeOptions.optimizeEmptyElementals = true;
  pm.addPass(hlfir::createBufferizeHLFIR(bufferizeOptions));
  // Run hlfir.assign inlining again after BufferizeHLFIR,
  // because the latter may introduce new hlfir.assign operations,
  // e.g. for copying an array into a temporary due to
  // hlfir.associate.
  // TODO: we can remove the previous InlineHLFIRAssign, when
  // FIR AliasAnalysis is good enough to say that a temporary
  // array does not alias with any user object.
  if (optLevel.isOptimizingForSpeed())
    addNestedPassToAllTopLevelOperations<PassConstructor>(
        pm, hlfir::createInlineHLFIRAssign);
  pm.addPass(hlfir::createConvertHLFIRtoFIR());
  if (enableOpenMP != EnableOpenMP::None)
    pm.addPass(flangomp::createLowerWorkshare());
  if (enableOpenMP == EnableOpenMP::Simd)
    pm.addPass(flangomp::createSimdOnlyPass());
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
void createOpenMPFIRPassPipeline(mlir::PassManager &pm,
                                 OpenMPFIRPassPipelineOpts opts) {
  using DoConcurrentMappingKind =
      Fortran::frontend::CodeGenOptions::DoConcurrentMappingKind;

  if (opts.doConcurrentMappingKind != DoConcurrentMappingKind::DCMK_None)
    pm.addPass(flangomp::createDoConcurrentConversionPass(
        opts.doConcurrentMappingKind == DoConcurrentMappingKind::DCMK_Device));

  // The MapsForPrivatizedSymbols and AutomapToTargetDataPass pass need to run
  // before MapInfoFinalizationPass because they create new MapInfoOp
  // instances, typically for descriptors. MapInfoFinalizationPass adds
  // MapInfoOp instances for the descriptors underlying data which is necessary
  // to access the data on the offload target device.
  pm.addPass(flangomp::createMapsForPrivatizedSymbolsPass());
  pm.addPass(flangomp::createAutomapToTargetDataPass());
  pm.addPass(flangomp::createMapInfoFinalizationPass());
  pm.addPass(flangomp::createMarkDeclareTargetPass());
  pm.addPass(flangomp::createGenericLoopConversionPass());
  if (opts.isTargetDevice)
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
  addNestedPassToAllTopLevelOperations<PassConstructor>(
      pm, fir::createAbstractResultOpt);
  addPassToGPUModuleOperations<PassConstructor>(pm,
                                                fir::createAbstractResultOpt);
  fir::addCodeGenRewritePass(
      pm, (config.DebugInfo != llvm::codegenoptions::NoDebugInfo));
  fir::addExternalNameConversionPass(pm, config.Underscoring);
  fir::createDebugPasses(pm, config.DebugInfo, config.OptLevel, inputFilename);
  fir::addTargetRewritePass(pm);
  fir::addCompilerGeneratedNamesConversionPass(pm);

  if (config.VScaleMin != 0)
    pm.addPass(fir::createVScaleAttr({{config.VScaleMin, config.VScaleMax}}));

  // Add function attributes
  mlir::LLVM::framePointerKind::FramePointerKind framePointerKind;

  if (config.FramePointerKind == llvm::FramePointerKind::NonLeaf)
    framePointerKind = mlir::LLVM::framePointerKind::FramePointerKind::NonLeaf;
  else if (config.FramePointerKind == llvm::FramePointerKind::All)
    framePointerKind = mlir::LLVM::framePointerKind::FramePointerKind::All;
  else if (config.FramePointerKind == llvm::FramePointerKind::Reserved)
    framePointerKind = mlir::LLVM::framePointerKind::FramePointerKind::Reserved;
  else
    framePointerKind = mlir::LLVM::framePointerKind::FramePointerKind::None;

  // TODO: re-enable setNoAlias by default (when optimizing for speed) once
  // function specialization is fixed.
  bool setNoAlias = forceNoAlias;
  bool setNoCapture = config.OptLevel.isOptimizingForSpeed();

  pm.addPass(fir::createFunctionAttr(
      {framePointerKind, config.InstrumentFunctionEntry,
       config.InstrumentFunctionExit, config.NoInfsFPMath, config.NoNaNsFPMath,
       config.ApproxFuncFPMath, config.NoSignedZerosFPMath, config.UnsafeFPMath,
       config.Reciprocals, config.PreferVectorWidth, /*tuneCPU=*/"",
       setNoCapture, setNoAlias}));

  if (config.EnableOpenMP) {
    pm.addNestedPass<mlir::func::FuncOp>(
        flangomp::createLowerNontemporalPass());
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
  fir::EnableOpenMP enableOpenMP = fir::EnableOpenMP::None;
  if (config.EnableOpenMP)
    enableOpenMP = fir::EnableOpenMP::Full;
  if (config.EnableOpenMPSimd)
    enableOpenMP = fir::EnableOpenMP::Simd;
  fir::createHLFIRToFIRPassPipeline(pm, enableOpenMP, config.OptLevel);

  // Add default optimizer pass pipeline.
  fir::createDefaultFIROptimizerPassPipeline(pm, config);

  // Add codegen pass pipeline.
  fir::createDefaultFIRCodeGenPassPipeline(pm, config, inputFilename);
}

} // namespace fir
