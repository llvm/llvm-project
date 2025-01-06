//===-- runtime/CUDA/memmove-function.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/memmove-function.h"
#include "../terminator.h"
#include "flang/Runtime/CUDA/common.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {

void *MemmoveHostToDevice(void *dst, const void *src, std::size_t count) {
  // TODO: Use cudaMemcpyAsync when we have support for stream.
  CUDA_REPORT_IF_ERROR(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
  return dst;
}

void *MemmoveDeviceToHost(void *dst, const void *src, std::size_t count) {
  // TODO: Use cudaMemcpyAsync when we have support for stream.
  CUDA_REPORT_IF_ERROR(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
  return dst;
}

void *MemmoveDeviceToDevice(void *dst, const void *src, std::size_t count) {
  // TODO: Use cudaMemcpyAsync when we have support for stream.
  CUDA_REPORT_IF_ERROR(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
  return dst;
}

} // namespace Fortran::runtime::cuda
