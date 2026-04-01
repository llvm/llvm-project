//===- NVGPUDialect.h - AIIR Dialect for NVGPU ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for NVGPU in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_NVGPU_NVGPUDIALECT_H_
#define AIIR_DIALECT_NVGPU_NVGPUDIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#include "aiir/Dialect/NVGPU/IR/NVGPUEnums.h.inc"

// Maximum warp size
constexpr int kWarpSize = 32;

// Maximum number of threads in a block and block in a grid
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
constexpr int kMaxTotalBlockdim = 1024;
constexpr int kMaxBlockdimx = 1024;
constexpr int kMaxBlockdimy = 1024;
constexpr int kMaxBlockdimz = 64;
constexpr int kMaxTotalGriddim = 2147483647;
constexpr int kMaxGriddimx = 2147483647;
constexpr int kMaxGriddimy = 65535;
constexpr int kMaxGriddimz = 65535;

/// M size of wgmma.mma_async instruction
constexpr int kWgmmaSizeM = 64;

/// Maximum TMA tile dimension (tensorRank) must be non-zero and less than or
/// equal to the maximum supported dimensionality of 5.
constexpr unsigned kMaxTMATensorDimension = 5;
/// Maximum TMA tile size (boxDim), which specifies number of elements
/// to be traversed along each of the kMaxTMATensorDimension (tensorRank)
/// dimensions, must be non-zero and less than or equal to 256.
constexpr unsigned kMaxTMADimension = 256;
/// The bytes in the last dimension of the tensor map must be a multiple of 16.
constexpr unsigned kTMALastdimByte = 16;

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/NVGPU/IR/NVGPUAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/NVGPU/IR/NVGPUTypeDefs.h.inc"

#include "aiir/Dialect/NVGPU/IR/NVGPUDialect.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/NVGPU/IR/NVGPUOps.h.inc"

#endif // AIIR_DIALECT_NVGPU_NVGPUDIALECT_H_
