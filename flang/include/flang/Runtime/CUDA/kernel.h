//===-- include/flang/Runtime/CUDA/kernel.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_KERNEL_H_
#define FORTRAN_RUNTIME_CUDA_KERNEL_H_

#include "flang/Runtime/entry-names.h"
#include <cstddef>
#include <stdint.h>

extern "C" {

// These functions use intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.

void RTDEF(CUFLaunchKernel)(const void *kernelName, intptr_t gridX,
    intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY,
    intptr_t blockZ, int32_t smem, void **params, void **extra);

void RTDEF(CUFLaunchClusterKernel)(const void *kernelName, intptr_t clusterX,
    intptr_t clusterY, intptr_t clusterZ, intptr_t gridX, intptr_t gridY,
    intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
    int32_t smem, void **params, void **extra);

} // extern "C"

#endif // FORTRAN_RUNTIME_CUDA_KERNEL_H_
