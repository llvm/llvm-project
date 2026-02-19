//===-- lib/cuda/stream.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/stream.h"
#include "flang-rt/runtime/allocator-registry.h"
#include "flang-rt/runtime/derived.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/lock.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Support/Fortran.h"

namespace Fortran::runtime::cuda {

static thread_local cudaStream_t defaultStream{nullptr};

extern "C" {

int RTDECL(CUFSetDefaultStream)(cudaStream_t stream) {
  defaultStream = stream;
  return StatOk;
}

cudaStream_t RTDECL(CUFGetDefaultStream)() { return defaultStream; }

int RTDECL(CUFStreamSynchronize)(cudaStream_t stream) {
  return cudaStreamSynchronize(stream);
}

int RTDECL(CUFStreamSynchronizeNull)() {
  cudaStream_t defaultStream = 0;
  return cudaStreamSynchronize(defaultStream);
}
}

} // namespace Fortran::runtime::cuda
