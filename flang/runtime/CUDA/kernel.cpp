//===-- runtime/CUDA/kernel.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/kernel.h"
#include "../terminator.h"
#include "flang/Runtime/CUDA/common.h"

#include "cuda_runtime.h"

extern "C" {

void RTDEF(CUFLaunchKernel)(const void *kernel, intptr_t gridX, intptr_t gridY,
    intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
    int32_t smem, void **params, void **extra) {
  dim3 gridDim;
  gridDim.x = gridX;
  gridDim.y = gridY;
  gridDim.z = gridZ;
  dim3 blockDim;
  blockDim.x = blockX;
  blockDim.y = blockY;
  blockDim.z = blockZ;
  cudaStream_t stream = 0;
  CUDA_REPORT_IF_ERROR(
      cudaLaunchKernel(kernel, gridDim, blockDim, params, smem, stream));
}

} // extern "C"
