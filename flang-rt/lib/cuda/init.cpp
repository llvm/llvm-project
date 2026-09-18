//===-- lib/cuda/init.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/init.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Runtime/CUDA/common.h"

#include "cuda_runtime.h"

extern "C" {

void RTDEF(CUFInit)() {
  // Perform ctx initialization based on execution environment if necessary.
  if (Fortran::runtime::executionEnvironment.cudaStackLimit) {
    CUDA_REPORT_IF_ERROR(cudaDeviceSetLimit(cudaLimitStackSize,
        Fortran::runtime::executionEnvironment.cudaStackLimit));
  }
  // Mirror the host execution environment (already configured: lowering
  // generates the CUFInit call after ProgramStart) into the device image's
  // copy, so that the environment variables documented in
  // flang/docs/RuntimeEnvironment.md have the same effect in runtime code
  // compiled for the device as they have on the host. Without this, the
  // device copy stays at its default-initialized values. When the program
  // links no device-side flang-rt, the symbol is not registered with the
  // CUDA runtime; treat that as nothing to sync.
  void *devicePtr{nullptr};
  if (cudaGetSymbolAddress(
          &devicePtr, &Fortran::runtime::executionEnvironment) == cudaSuccess) {
    CUDA_REPORT_IF_ERROR(
        cudaMemcpy(devicePtr, &Fortran::runtime::executionEnvironment,
            sizeof(Fortran::runtime::executionEnvironment),
            cudaMemcpyHostToDevice));
  } else {
    // Clear the sticky cudaErrorInvalidSymbol.
    (void)cudaGetLastError();
  }
}
}
