//===-- include/flang/Runtime/CUDA/init.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_INIT_H_
#define FORTRAN_RUNTIME_CUDA_INIT_H_

#include "flang/Runtime/entry-names.h"

extern "C" {

void RTDECL(CUFInit)();

// Mirrors the host executionEnvironment (sanitized: no host argc/argv/envp)
// into the device image's copy. Called by CUFInit; a no-op unless Flang-RT
// was built with device offload support. Host-only (bare RTNAME): the
// implementation launches a kernel and synchronizes, so it must not be
// compiled __host__ __device__ in the CUDA translation unit defining it.
void RTNAME(CUFSyncExecutionEnvironment)();
}

#endif // FORTRAN_RUNTIME_CUDA_INIT_H_
