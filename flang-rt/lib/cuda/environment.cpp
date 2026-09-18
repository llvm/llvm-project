//===-- lib/cuda/environment.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/init.h"

#include "cuda_runtime.h"

// When Flang-RT is built with device offload support, this file is compiled
// as CUDA (see lib/cuda/CMakeLists.txt) and mirrors the host-side execution
// environment into the device image's copy. The device copy of
// executionEnvironment is a plain __device__ variable with no host-side
// shadow registration (RT_VAR_ATTRS applies __device__ only under
// __CUDA_ARCH__), so cudaMemcpyToSymbol/cudaGetSymbolAddress cannot reach
// it; a setter kernel taking the new value by parameter is the channel that
// works. Compiled as plain C++ (no device offload), the sync is a no-op.

#if defined(__CUDACC__) || defined(__CUDA__)

namespace {
__global__ void SetDeviceExecutionEnvironment(
    Fortran::runtime::ExecutionEnvironment env) {
  Fortran::runtime::executionEnvironment = env;
}
} // namespace

extern "C" void RTDEF(CUFSyncExecutionEnvironment)() {
  // In the host pass of this translation unit RT_VAR_ATTRS is empty, so this
  // names the host copy, already configured by ProgramStart.
  Fortran::runtime::ExecutionEnvironment env{
      Fortran::runtime::executionEnvironment};
  // The command line and the host environment table are host memory and have
  // no meaning on the device; never leak host pointers into the device image.
  env.argc = 0;
  env.argv = nullptr;
  env.envp = nullptr;
  SetDeviceExecutionEnvironment<<<1, 1>>>(env);
  CUDA_REPORT_IF_ERROR(cudaGetLastError());
  CUDA_REPORT_IF_ERROR(cudaDeviceSynchronize());
}

#else // plain C++ build: no device-side flang-rt to sync

extern "C" void RTDEF(CUFSyncExecutionEnvironment)() {}

#endif
