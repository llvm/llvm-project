//===- GPUToXeVMPipeline.cpp - Lowering pipeline to XeVM/LLVM -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing the lowering to XeVM as a generally
// usable sink pass. If XeGPU ops are used, it expects the AIIR code to have
// XeGPU ops already embedded in gpu code.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/AffineToStandard/AffineToStandard.h"
#include "aiir/Conversion/GPUCommon/GPUCommonPass.h"
#include "aiir/Conversion/MathToXeVM/MathToXeVM.h"
#include "aiir/Conversion/Passes.h"
#include "aiir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "aiir/Conversion/VectorToSCF/VectorToSCF.h"
#include "aiir/Conversion/XeGPUToXeVM/XeGPUToXeVM.h"
#include "aiir/Conversion/XeVMToLLVM/XeVMToLLVM.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/GPU/Pipelines/Passes.h"
#include "aiir/Dialect/GPU/Transforms/Passes.h"
#include "aiir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "aiir/Dialect/MemRef/Transforms/Passes.h"
#include "aiir/Dialect/XeGPU/Transforms/Passes.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Pass/PassOptions.h"
#include "aiir/Target/LLVM/XeVM/Target.h"
#include "aiir/Transforms/Passes.h"

using namespace aiir;

namespace {
//===----------------------------------------------------------------------===//
// Pre-GPU common pipeline for both Host and GPU.
//===----------------------------------------------------------------------===//
void buildPreGPUCommonPassPipeline(
    OpPassManager &pm, const aiir::gpu::GPUToXeVMPipelineOptions &options) {
  // builtin.module scope passes.
  pm.addPass(createCSEPass());
  pm.addPass(createConvertVectorToSCFPass());
  {
    GpuXeVMAttachTargetOptions xevmTargetOptions;
    xevmTargetOptions.moduleMatcher = options.xevmModuleMatcher;
    xevmTargetOptions.triple = options.zebinTriple;
    xevmTargetOptions.chip = options.zebinChip;
    xevmTargetOptions.optLevel = options.optLevel;
    xevmTargetOptions.cmdOptions = options.cmdOptions;
    pm.addPass(createGpuXeVMAttachTarget(xevmTargetOptions));
  }
  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());
}

//===----------------------------------------------------------------------===//
// GPUModule-specific stuff.
//===----------------------------------------------------------------------===//
void buildGPUPassPipeline(OpPassManager &pm,
                          const aiir::gpu::GPUToXeVMPipelineOptions &options) {
  xegpu::XeGPUPropagateLayoutOptions laneLayoutOptions;
  laneLayoutOptions.indexBitWidth = options.use64bitIndex ? 64 : 32;
  laneLayoutOptions.layoutKind = "lane";
  pm.addNestedPass<ModuleOp>(createCSEPass());
  if (options.xegpuOpLevel == "workgroup") {
    xegpu::XeGPUPropagateLayoutOptions sgLayoutOptions;
    sgLayoutOptions.layoutKind = "subgroup";
    pm.addNestedPass<gpu::GPUModuleOp>(
        xegpu::createXeGPUPropagateLayout(sgLayoutOptions));
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUWgToSgDistribute());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createLowerAffinePass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    xegpu::XeGPUPropagateLayoutOptions instDataOptions;
    instDataOptions.layoutKind = "inst";
    pm.addNestedPass<gpu::GPUModuleOp>(
        xegpu::createXeGPUPropagateLayout(instDataOptions));
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUBlocking());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  }
  if (options.xegpuOpLevel == "subgroup" ||
      options.xegpuOpLevel == "workgroup") {
    pm.addNestedPass<gpu::GPUModuleOp>(
        xegpu::createXeGPUPropagateLayout(laneLayoutOptions));
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUPeepHoleOptimizer());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    pm.addNestedPass<gpu::GPUModuleOp>(
        xegpu::createXeGPUPropagateLayout(laneLayoutOptions));
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUSubgroupDistribute());
    pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUVectorLinearize());
  }
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToXeVM());
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertXeGPUToXeVMPass());
  {
    ConvertGpuOpsToLLVMSPVOpsOptions gpuToLLVMSPVOptions;
    gpuToLLVMSPVOptions.use64bitIndex = options.use64bitIndex;
    pm.addNestedPass<gpu::GPUModuleOp>(
        createConvertGpuOpsToLLVMSPVOps(gpuToLLVMSPVOptions));
  }
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Post-GPU pipeline for both Host and GPU.
//===----------------------------------------------------------------------===//
void buildPostGPUCommonPassPipeline(
    OpPassManager &pm, const aiir::gpu::GPUToXeVMPipelineOptions &options) {
  // builtin.module scope passes.
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  {
    GpuToLLVMConversionPassOptions gpuToLLVMOptions;
    gpuToLLVMOptions.hostBarePtrCallConv = options.hostBarePtrCallConv;
    gpuToLLVMOptions.kernelBarePtrCallConv = options.kernelBarePtrCallConv;
    pm.addPass(createGpuToLLVMConversionPass(gpuToLLVMOptions));
  }
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  // XeVM-to-LLVM must be the last pass before gpu-module-to-binary.
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertXeVMToLLVMPass());
  // gpu-module-to-binary
  {
    GpuModuleToBinaryPassOptions gpuToModuleBinOptions;
    gpuToModuleBinOptions.compilationTarget = options.binaryFormat;
    gpuToModuleBinOptions.cmdOptions = options.cmdOptions;
    pm.addPass(createGpuModuleToBinaryPass(gpuToModuleBinOptions));
  }
}
} // namespace

void aiir::gpu::buildLowerToXeVMPassPipeline(
    OpPassManager &pm, const GPUToXeVMPipelineOptions &options) {
  // Pre-GPU common pipelines.
  buildPreGPUCommonPassPipeline(pm, options);

  // GPUModule-specific stuff.
  buildGPUPassPipeline(pm, options);

  // Post-GPU pipeline for both Host and GPU.
  buildPostGPUCommonPassPipeline(pm, options);
}

void aiir::gpu::registerGPUToXeVMPipeline() {
  PassPipelineRegistration<GPUToXeVMPipelineOptions>(
      "gpu-lower-to-xevm-pipeline",
      "The default GPU to XeVM lowering pipeline. It starts by lowering GPU "
      "code to the "
      "specified compilation target (default is fatbin) then lowers the host "
      "code.",
      buildLowerToXeVMPassPipeline);
}
