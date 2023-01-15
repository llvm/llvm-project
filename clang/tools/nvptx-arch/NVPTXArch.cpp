//===- NVPTXArch.cpp - list installed NVPTX devies ------*- C++ -*---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool for detecting name of CUDA gpus installed in the
// system.
//
//===----------------------------------------------------------------------===//

#if defined(__has_include)
#if __has_include("cuda.h")
#include "cuda.h"
#define CUDA_HEADER_FOUND 1
#else
#define CUDA_HEADER_FOUND 0
#endif
#else
#define CUDA_HEADER_FOUND 0
#endif

#if !CUDA_HEADER_FOUND
int main() { return 1; }
#else

#include <cstdint>
#include <cstdio>

static int handleError(CUresult Err) {
  const char *ErrStr = nullptr;
  CUresult Result = cuGetErrorString(Err, &ErrStr);
  if (Result != CUDA_SUCCESS)
    return EXIT_FAILURE;
  fprintf(stderr, "CUDA error: %s\n", ErrStr);
  return EXIT_FAILURE;
}

int main() {
  if (CUresult Err = cuInit(0)) {
    if (Err == CUDA_ERROR_NO_DEVICE)
      return EXIT_SUCCESS;
    else
      return handleError(Err);
  }

  int Count = 0;
  if (CUresult Err = cuDeviceGetCount(&Count))
    return handleError(Err);
  if (Count == 0)
    return EXIT_SUCCESS;
  for (int DeviceId = 0; DeviceId < Count; ++DeviceId) {
    CUdevice Device;
    if (CUresult Err = cuDeviceGet(&Device, DeviceId))
      return handleError(Err);

    int32_t Major, Minor;
    if (CUresult Err = cuDeviceGetAttribute(
            &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Device))
      return handleError(Err);
    if (CUresult Err = cuDeviceGetAttribute(
            &Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Device))
      return handleError(Err);

    printf("sm_%d%d\n", Major, Minor);
  }
  return EXIT_SUCCESS;
}

#endif
