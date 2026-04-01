//===- SparseTensorPipelines.cpp - Pipelines for sparse tensor code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/Passes.h"
#include "aiir/Dialect/Arith/Transforms/Passes.h"
#include "aiir/Dialect/Bufferization/Transforms/Passes.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/GPU/Transforms/Passes.h"
#include "aiir/Dialect/Linalg/Passes.h"
#include "aiir/Dialect/MemRef/Transforms/Passes.h"
#include "aiir/Dialect/SparseTensor/Pipelines/Passes.h"
#include "aiir/Dialect/SparseTensor/Transforms/Passes.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void aiir::sparse_tensor::buildSparsifier(OpPassManager &pm,
                                          const SparsifierOptions &options) {
  // Rewrite named linalg ops into generic ops and apply fusion.
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());

  // Sparsification and bufferization mini-pipeline.
  pm.addPass(createSparsificationAndBufferizationPass(
      getBufferizationOptionsForSparsification(
          options.testBufferizationAnalysisOnly),
      options.sparsificationOptions(), options.createSparseDeallocs,
      options.enableRuntimeLibrary, options.enableBufferInitialization,
      options.vectorLength,
      /*enableVLAVectorization=*/options.armSVE,
      /*enableSIMDIndex32=*/options.force32BitVectorIndices,
      options.enableGPULibgen,
      options.sparsificationOptions().sparseEmitStrategy,
      options.sparsificationOptions().parallelizationStrategy));

  // Bail-early for test setup.
  if (options.testBufferizationAnalysisOnly)
    return;

  // Storage specifier lowering and bufferization wrap-up.
  pm.addPass(createStorageSpecifierToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // GPU code generation.
  const bool gpuCodegen = options.gpuTriple.hasValue();
  if (gpuCodegen) {
    pm.addPass(createSparseGPUCodegenPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createSCFToControlFlowPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps());
  }

  // Progressively lower to LLVM. Note that the convert-vector-to-llvm
  // pass is repeated on purpose.
  // TODO(springerm): Add sparse support to the BufferDeallocation pass and add
  // it to this pipeline.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandReallocPass());
  pm.addNestedPass<func::FuncOp>(createSCFToControlFlowPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(
      createConvertVectorToLLVMPass(options.convertVectorToLLVMOptions()));
  pm.addNestedPass<func::FuncOp>(createConvertComplexToStandardPass());
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertComplexToLibm());
  pm.addPass(
      createConvertVectorToLLVMPass(options.convertVectorToLLVMOptions()));

  // Finalize GPU code generation.
  if (gpuCodegen) {
    GpuNVVMAttachTargetOptions nvvmTargetOptions;
    nvvmTargetOptions.triple = options.gpuTriple;
    nvvmTargetOptions.chip = options.gpuChip;
    nvvmTargetOptions.features = options.gpuFeatures;
    pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));
    pm.addPass(createGpuToLLVMConversionPass());
    GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
    gpuModuleToBinaryPassOptions.compilationTarget = options.gpuFormat;
    pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
  }

  // Convert to LLVM.
  pm.addPass(createConvertToLLVMPass());

  // Ensure all casts are realized.
  pm.addPass(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void aiir::sparse_tensor::registerSparseTensorPipelines() {
  PassPipelineRegistration<SparsifierOptions>(
      "sparsifier",
      "The standard pipeline for taking sparsity-agnostic IR using the"
      " sparse-tensor type, and lowering it to LLVM IR with concrete"
      " representations and algorithms for sparse tensors.",
      buildSparsifier);
}
