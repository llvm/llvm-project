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
}
}
