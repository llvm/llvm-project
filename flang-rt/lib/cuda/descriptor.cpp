//===-- lib/cuda/descriptor.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/descriptor.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Runtime/CUDA/allocator.h"
#include "flang/Runtime/CUDA/common.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {
extern "C" {
RT_EXT_API_GROUP_BEGIN

Descriptor *RTDEF(CUFAllocDescriptor)(
    std::size_t sizeInBytes, const char *sourceFile, int sourceLine) {
  return reinterpret_cast<Descriptor *>(
      CUFAllocManaged(sizeInBytes, /*asyncObject=*/nullptr));
}

void RTDEF(CUFFreeDescriptor)(
    Descriptor *desc, const char *sourceFile, int sourceLine) {
  CUFFreeManaged(reinterpret_cast<void *>(desc));
}

void *RTDEF(CUFGetDeviceAddress)(
    void *hostPtr, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  void *p;
  CUDA_REPORT_IF_ERROR(cudaGetSymbolAddress((void **)&p, hostPtr));
  if (!p) {
    terminator.Crash("Could not retrieve symbol's address");
  }
  return p;
}

void RTDEF(CUFDescriptorSync)(Descriptor *dst, const Descriptor *src,
    const char *sourceFile, int sourceLine) {
  std::size_t count{src->SizeInBytes()};
  CUDA_REPORT_IF_ERROR(cudaMemcpy(
      (void *)dst, (const void *)src, count, cudaMemcpyHostToDevice));
}

void RTDEF(CUFSyncGlobalDescriptor)(
    void *hostPtr, const char *sourceFile, int sourceLine) {
  void *devAddr{RTNAME(CUFGetDeviceAddress)(hostPtr, sourceFile, sourceLine)};
  RTNAME(CUFDescriptorSync)
  ((Descriptor *)devAddr, (Descriptor *)hostPtr, sourceFile, sourceLine);
}

void RTDEF(CUFDescriptorCheckSection)(
    const Descriptor *desc, const char *sourceFile, int sourceLine) {
  if (desc && !desc->IsContiguous()) {
    Terminator terminator{sourceFile, sourceLine};
    terminator.Crash("device array section argument is not contiguous");
  }
}

RT_EXT_API_GROUP_END
}
} // namespace Fortran::runtime::cuda
