//===- RegisterAllExtensions.cpp - AIIR Extension Registration --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialect
// extensions to the system.
//
//===----------------------------------------------------------------------===//

#include "aiir/InitAllExtensions.h"

#include "aiir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "aiir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "aiir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "aiir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "aiir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "aiir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "aiir/Conversion/GPUCommon/GPUToLLVM.h"
#include "aiir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "aiir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "aiir/Conversion/MPIToLLVM/MPIToLLVM.h"
#include "aiir/Conversion/MathToLLVM/MathToLLVM.h"
#include "aiir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "aiir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "aiir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "aiir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "aiir/Conversion/PtrToLLVM/PtrToLLVM.h"
#include "aiir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "aiir/Conversion/UBToLLVM/UBToLLVM.h"
#include "aiir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "aiir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "aiir/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.h"
#include "aiir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.h"
#include "aiir/Dialect/Bufferization/Extensions/AllExtensions.h"
#include "aiir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "aiir/Dialect/DLTI/TransformOps/DLTITransformOps.h"
#include "aiir/Dialect/Func/Extensions/AllExtensions.h"
#include "aiir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "aiir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "aiir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "aiir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "aiir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "aiir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "aiir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.h"
#include "aiir/Dialect/Tensor/Extensions/AllExtensions.h"
#include "aiir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "aiir/Dialect/Transform/DebugExtension/DebugExtension.h"
#include "aiir/Dialect/Transform/IRDLExtension/IRDLExtension.h"
#include "aiir/Dialect/Transform/LoopExtension/LoopExtension.h"
#include "aiir/Dialect/Transform/PDLExtension/PDLExtension.h"
#include "aiir/Dialect/Transform/SMTExtension/SMTExtension.h"
#include "aiir/Dialect/Transform/TuneExtension/TuneExtension.h"
#include "aiir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "aiir/Dialect/X86/TransformOps/X86TransformOps.h"
#include "aiir/Dialect/X86/Transforms.h"
#include "aiir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.h"
#include "aiir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"

/// This function may be called to register all AIIR dialect extensions with the
/// provided registry.
/// If you're building a compiler, you generally shouldn't use this: you would
/// individually register the specific extensions that are useful for the
/// pipelines and transformations you are using.
void aiir::registerAllExtensions(DialectRegistry &registry) {
  // Register all conversions to LLVM extensions.
  registerConvertArithToEmitCInterface(registry);
  arith::registerConvertArithToLLVMInterface(registry);
  bufferization::registerAllExtensions(registry);
  registerConvertComplexToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  func::registerAllExtensions(registry);
  tensor::registerAllExtensions(registry);
  registerConvertFuncToEmitCInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  mpi::registerConvertMPIToLLVMInterface(registry);
  registerConvertMemRefToEmitCInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  registerConvertNVVMToLLVMInterface(registry);
  ptr::registerConvertPtrToLLVMInterface(registry);
  registerConvertOpenMPToLLVMInterface(registry);
  registerConvertSCFToEmitCInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);
  gpu::registerConvertGpuToLLVMInterface(registry);
  NVVM::registerConvertGpuToNVVMInterface(registry);
  vector::registerConvertVectorToLLVMInterface(registry);
  registerConvertX86ToLLVMInterface(registry);

  // Register all transform dialect extensions.
  affine::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  dlti::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  gpu::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  nvgpu::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  sparse_tensor::registerTransformDialectExtension(registry);
  tensor::registerTransformDialectExtension(registry);
  transform::registerDebugExtension(registry);
  transform::registerIRDLExtension(registry);
  transform::registerLoopExtension(registry);
  transform::registerPDLExtension(registry);
  transform::registerSMTExtension(registry);
  transform::registerTuneExtension(registry);
  vector::registerTransformDialectExtension(registry);
  x86::registerTransformDialectExtension(registry);
  xegpu::registerTransformDialectExtension(registry);
  arm_neon::registerTransformDialectExtension(registry);
  arm_sve::registerTransformDialectExtension(registry);

  // Translation extensions need to be registered by calling
  // `registerAllToLLVMIRTranslations` (see All.h).
}
