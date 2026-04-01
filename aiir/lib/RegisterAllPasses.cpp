//===- RegisterAllPasses.cpp - AIIR Registration ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes to the
// system.
//
//===----------------------------------------------------------------------===//

#include "aiir/InitAllPasses.h"

#include "aiir/Conversion/Passes.h"
#include "aiir/Dialect/AMDGPU/Transforms/Passes.h"
#include "aiir/Dialect/Affine/Transforms/Passes.h"
#include "aiir/Dialect/Arith/Transforms/Passes.h"
#include "aiir/Dialect/ArmSME/Transforms/Passes.h"
#include "aiir/Dialect/ArmSVE/Transforms/Passes.h"
#include "aiir/Dialect/Async/Passes.h"
#include "aiir/Dialect/Bufferization/Pipelines/Passes.h"
#include "aiir/Dialect/Bufferization/Transforms/Passes.h"
#include "aiir/Dialect/EmitC/Transforms/Passes.h"
#include "aiir/Dialect/Func/Transforms/Passes.h"
#include "aiir/Dialect/GPU/Pipelines/Passes.h"
#include "aiir/Dialect/GPU/Transforms/Passes.h"
#include "aiir/Dialect/LLVMIR/Transforms/Passes.h"
#include "aiir/Dialect/Linalg/Passes.h"
#include "aiir/Dialect/MLProgram/Transforms/Passes.h"
#include "aiir/Dialect/Math/Transforms/Passes.h"
#include "aiir/Dialect/MemRef/Transforms/Passes.h"
#include "aiir/Dialect/NVGPU/Transforms/Passes.h"
#include "aiir/Dialect/OpenACC/Transforms/Passes.h"
#include "aiir/Dialect/OpenMP/Transforms/Passes.h"
#include "aiir/Dialect/Quant/Transforms/Passes.h"
#include "aiir/Dialect/SCF/Transforms/Passes.h"
#include "aiir/Dialect/SPIRV/Transforms/Passes.h"
#include "aiir/Dialect/Shape/Transforms/Passes.h"
#include "aiir/Dialect/Shard/Transforms/Passes.h"
#include "aiir/Dialect/SparseTensor/Pipelines/Passes.h"
#include "aiir/Dialect/SparseTensor/Transforms/Passes.h"
#include "aiir/Dialect/Tensor/Transforms/Passes.h"
#include "aiir/Dialect/Tosa/Transforms/Passes.h"
#include "aiir/Dialect/Transform/Transforms/Passes.h"
#include "aiir/Dialect/Vector/Transforms/Passes.h"
#include "aiir/Dialect/XeGPU/Transforms/Passes.h"
#include "aiir/Target/LLVMIR/Transforms/Passes.h"
#include "aiir/Transforms/Passes.h"

// This function may be called to register the AIIR passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
void aiir::registerAllPasses() {
  // General passes
  registerTransformsPasses();

  // Conversion passes
  registerConversionPasses();

  // Dialect passes
  acc::registerOpenACCPasses();
  affine::registerAffinePasses();
  amdgpu::registerAMDGPUPasses();
  registerAsyncPasses();
  arith::registerArithPasses();
  bufferization::registerBufferizationPasses();
  func::registerFuncPasses();
  registerGPUPasses();
  registerLinalgPasses();
  registerNVGPUPasses();
  registerSparseTensorPasses();
  LLVM::registerLLVMPasses();
  LLVM::registerTargetLLVMIRTransformsPasses();
  math::registerMathPasses();
  memref::registerMemRefPasses();
  shard::registerShardPasses();
  ml_program::registerMLProgramPasses();
  omp::registerOpenMPPasses();
  quant::registerQuantPasses();
  registerSCFPasses();
  registerShapePasses();
  spirv::registerSPIRVPasses();
  tensor::registerTensorPasses();
  tosa::registerTosaPasses();
  transform::registerTransformPasses();
  vector::registerVectorPasses();
  arm_sme::registerArmSMEPasses();
  arm_sve::registerArmSVEPasses();
  emitc::registerEmitCPasses();
  xegpu::registerXeGPUPasses();

  // Dialect pipelines
  bufferization::registerBufferizationPipelines();
  sparse_tensor::registerSparseTensorPipelines();
  tosa::registerTosaToLinalgPipelines();
  gpu::registerGPUToNVVMPipeline();
  gpu::registerGPUToXeVMPipeline();
}
