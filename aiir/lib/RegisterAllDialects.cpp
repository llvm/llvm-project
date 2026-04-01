//===- RegisterAllDialects.cpp - AIIR Dialects Registration -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#include "aiir/InitAllDialects.h"

#include "aiir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "aiir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "aiir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "aiir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/Arith/Transforms/ShardingInterfaceImpl.h"
#include "aiir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "aiir/Dialect/ArmSME/IR/ArmSME.h"
#include "aiir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "aiir/Dialect/Async/IR/Async.h"
#include "aiir/Dialect/Bufferization/IR/Bufferization.h"
#include "aiir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/Complex/IR/Complex.h"
#include "aiir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "aiir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "aiir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/EmitC/IR/EmitC.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/GPU/IR/ValueBoundsOpInterfaceImpl.h"
#include "aiir/Dialect/GPU/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "aiir/Dialect/IRDL/IR/IRDL.h"
#include "aiir/Dialect/Index/IR/IndexDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/LLVMIR/NVVMDialect.h"
#include "aiir/Dialect/LLVMIR/ROCDLDialect.h"
#include "aiir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "aiir/Dialect/LLVMIR/XeVMDialect.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "aiir/Dialect/Linalg/Transforms/RuntimeOpVerification.h"
#include "aiir/Dialect/MLProgram/IR/MLProgram.h"
#include "aiir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/MPI/IR/MPI.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "aiir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "aiir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "aiir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "aiir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "aiir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/Dialect/PDL/IR/PDL.h"
#include "aiir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "aiir/Dialect/Ptr/IR/PtrDialect.h"
#include "aiir/Dialect/Quant/IR/Quant.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "aiir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "aiir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "aiir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/SMT/IR/SMTDialect.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/Shape/IR/Shape.h"
#include "aiir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/Shard/IR/ShardDialect.h"
#include "aiir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "aiir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "aiir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "aiir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "aiir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "aiir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/Tensor/Transforms/RuntimeOpVerification.h"
#include "aiir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "aiir/Dialect/Tosa/IR/ShardingInterfaceImpl.h"
#include "aiir/Dialect/Tosa/IR/TosaOps.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/PDLExtension/PDLExtension.h"
#include "aiir/Dialect/UB/IR/UBOps.h"
#include "aiir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "aiir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "aiir/Dialect/X86/X86Dialect.h"
#include "aiir/Dialect/XeGPU/IR/XeGPU.h"
#include "aiir/IR/Dialect.h"
#include "aiir/Interfaces/CastInterfaces.h"
#include "aiir/Target/LLVM/NVVM/Target.h"
#include "aiir/Target/LLVM/ROCDL/Target.h"
#include "aiir/Target/LLVM/XeVM/Target.h"
#include "aiir/Target/SPIRV/Target.h"

/// Add all the AIIR dialects to the provided registry.
void aiir::registerAllDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<acc::OpenACCDialect,
                  affine::AffineDialect,
                  amdgpu::AMDGPUDialect,
                  arith::ArithDialect,
                  arm_neon::ArmNeonDialect,
                  arm_sme::ArmSMEDialect,
                  arm_sve::ArmSVEDialect,
                  async::AsyncDialect,
                  bufferization::BufferizationDialect,
                  cf::ControlFlowDialect,
                  complex::ComplexDialect,
                  DLTIDialect,
                  emitc::EmitCDialect,
                  func::FuncDialect,
                  gpu::GPUDialect,
                  index::IndexDialect,
                  irdl::IRDLDialect,
                  linalg::LinalgDialect,
                  LLVM::LLVMDialect,
                  math::MathDialect,
                  memref::MemRefDialect,
                  shard::ShardDialect,
                  ml_program::MLProgramDialect,
                  mpi::MPIDialect,
                  nvgpu::NVGPUDialect,
                  NVVM::NVVMDialect,
                  omp::OpenMPDialect,
                  pdl::PDLDialect,
                  pdl_interp::PDLInterpDialect,
                  ptr::PtrDialect,
                  quant::QuantDialect,
                  ROCDL::ROCDLDialect,
                  scf::SCFDialect,
                  shape::ShapeDialect,
                  smt::SMTDialect,
                  sparse_tensor::SparseTensorDialect,
                  spirv::SPIRVDialect,
                  tensor::TensorDialect,
                  tosa::TosaDialect,
                  transform::TransformDialect,
                  ub::UBDialect,
                  vector::VectorDialect,
                  wasmssa::WasmSSADialect,
                  x86::X86Dialect,
                  xegpu::XeGPUDialect,
                  xevm::XeVMDialect>();
  // clang-format on

  // Register all external models.
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  arith::registerShardingInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  builtin::registerCastOpInterfaceExternalModels(registry);
  cf::registerBufferizableOpInterfaceExternalModels(registry);
  cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);
  gpu::registerValueBoundsOpInterfaceExternalModels(registry);
  LLVM::registerInlinerInterface(registry);
  NVVM::registerInlinerInterface(registry);
  linalg::registerAllDialectInterfaceImplementations(registry);
  linalg::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  memref::registerAllocationOpInterfaceExternalModels(registry);
  memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  memref::registerValueBoundsOpInterfaceExternalModels(registry);
  memref::registerMemorySlotExternalModels(registry);
  ml_program::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  shape::registerBufferizableOpInterfaceExternalModels(registry);
  sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerFindPayloadReplacementOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  tosa::registerShardingInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerSubsetOpInterfaceExternalModels(registry);
  vector::registerValueBoundsOpInterfaceExternalModels(registry);
  NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  ROCDL::registerROCDLTargetInterfaceExternalModels(registry);
  spirv::registerSPIRVTargetInterfaceExternalModels(registry);
  xevm::registerXeVMTargetInterfaceExternalModels(registry);
}

/// Append all the AIIR dialects to the registry contained in the given context.
void aiir::registerAllDialects(AIIRContext &context) {
  DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}
