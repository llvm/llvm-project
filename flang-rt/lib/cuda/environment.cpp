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
// environment into the device image's copy through a setter kernel taking
// the new value by parameter. A setter kernel is used rather than the CUDA
// runtime symbol APIs because host-side shadow registration of the device
// executionEnvironment is toolchain-dependent: RT_VAR_ATTRS applies
// __device__ only under __CUDA_ARCH__, so the host compilation pass sees a
// plain host global. nvcc still emits a registration for the variable (its
// generated host stubs are derived from the device compilation), but clang
// derives registration from host-pass attributes and emits none, so
// cudaMemcpyToSymbol/cudaGetSymbolAddress would work under nvcc and fail
// under clang. The kernel-parameter channel needs no registration and works
// under both. Compiled as plain C++ (no device offload), the sync is a
// no-op.
//
// The entry point is deliberately host-only (bare RTNAME, not RTDEF): its
// body launches a kernel and synchronizes, which are host-only operations,
// and RTDEF would make it __host__ __device__ in this CUDA translation
// unit (a hard error under clang CUDA).

#if defined(__CUDACC__) || defined(__CUDA__)

namespace {
__global__ void SetDeviceExecutionEnvironment(
    Fortran::runtime::ExecutionEnvironment env) {
  Fortran::runtime::executionEnvironment = env;
}
} // namespace

extern "C" void RTNAME(CUFSyncExecutionEnvironment)() {
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

extern "C" void RTNAME(CUFSyncExecutionEnvironment)() {}

#endif
