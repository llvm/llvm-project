//===- GPUToNVVMPipeline.cpp - Test lowering to NVVM as a sink pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing the lowering to NVVM as a generally
// usable sink pass.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/AffineToStandard/AffineToStandard.h"
#include "aiir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "aiir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "aiir/Conversion/GPUCommon/GPUCommonPass.h"
#include "aiir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "aiir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "aiir/Conversion/MathToLLVM/MathToLLVM.h"
#include "aiir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "aiir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "aiir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "aiir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "aiir/Conversion/VectorToSCF/VectorToSCF.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/GPU/Pipelines/Passes.h"
#include "aiir/Dialect/GPU/Transforms/Passes.h"
#include "aiir/Dialect/MemRef/Transforms/Passes.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Pass/PassOptions.h"
#include "aiir/Transforms/Passes.h"

using namespace aiir;

namespace {

//===----------------------------------------------------------------------===//
// Common pipeline
//===----------------------------------------------------------------------===//
void buildCommonPassPipeline(
    OpPassManager &pm, const aiir::gpu::GPUToNVVMPipelineOptions &options) {
  pm.addPass(createConvertNVGPUToNVVMPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertNVVMToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());

  GpuNVVMAttachTargetOptions nvvmTargetOptions;
  nvvmTargetOptions.triple = options.cubinTriple;
  nvvmTargetOptions.chip = options.cubinChip;
  nvvmTargetOptions.features = options.cubinFeatures;
  nvvmTargetOptions.optLevel = options.optLevel;
  nvvmTargetOptions.cmdOptions = options.cmdOptions;
  pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
  convertIndexToLLVMPassOpt.indexBitwidth = options.indexBitWidth;
  pm.addPass(createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

//===----------------------------------------------------------------------===//
// GPUModule-specific stuff.
//===----------------------------------------------------------------------===//
void buildGpuPassPipeline(OpPassManager &pm,
                          const aiir::gpu::GPUToNVVMPipelineOptions &options) {
  ConvertGpuOpsToNVVMOpsOptions opt;
  opt.useBarePtrCallConv = options.kernelUseBarePtrCallConv;
  opt.indexBitwidth = options.indexBitWidth;
  opt.allowPatternRollback = options.allowPatternRollback;
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps(opt));
  pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Host Post-GPU pipeline
//===----------------------------------------------------------------------===//
void buildHostPostPipeline(OpPassManager &pm,
                           const aiir::gpu::GPUToNVVMPipelineOptions &options) {
  GpuToLLVMConversionPassOptions opt;
  opt.hostBarePtrCallConv = options.hostUseBarePtrCallConv;
  opt.kernelBarePtrCallConv = options.kernelUseBarePtrCallConv;
  pm.addPass(createGpuToLLVMConversionPass(opt));

  GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
  gpuModuleToBinaryPassOptions.compilationTarget = options.cubinFormat;
  pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace

void aiir::gpu::buildLowerToNVVMPassPipeline(
    OpPassManager &pm, const GPUToNVVMPipelineOptions &options) {
  // Common pipelines
  buildCommonPassPipeline(pm, options);

  // GPUModule-specific stuff
  buildGpuPassPipeline(pm, options);

  // Host post-GPUModule-specific stuff
  buildHostPostPipeline(pm, options);
}

void aiir::gpu::registerGPUToNVVMPipeline() {
  PassPipelineRegistration<GPUToNVVMPipelineOptions>(
      "gpu-lower-to-nvvm-pipeline",
      "The default pipeline lowers main dialects (arith, memref, scf, "
      "vector, gpu, and nvgpu) to NVVM. It starts by lowering GPU code to the "
      "specified compilation target (default is fatbin) then lowers the host "
      "code.",
      buildLowerToNVVMPassPipeline);
}
