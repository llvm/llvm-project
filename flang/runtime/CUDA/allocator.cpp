//===-- runtime/CUDA/allocator.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/allocator.h"
#include "../derived.h"
#include "../stat.h"
#include "../terminator.h"
#include "../type-info.h"
#include "flang/Common/Fortran.h"
#include "flang/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/allocator-registry.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {
extern "C" {

void RTDEF(CUFRegisterAllocator)() {
  allocatorRegistry.Register(
      kPinnedAllocatorPos, {&CUFAllocPinned, CUFFreePinned});
  allocatorRegistry.Register(
      kDeviceAllocatorPos, {&CUFAllocDevice, CUFFreeDevice});
  allocatorRegistry.Register(
      kManagedAllocatorPos, {&CUFAllocManaged, CUFFreeManaged});
  allocatorRegistry.Register(
      kUnifiedAllocatorPos, {&CUFAllocUnified, CUFFreeUnified});
}
}

void *CUFAllocPinned(std::size_t sizeInBytes) {
  void *p;
  CUDA_REPORT_IF_ERROR(cudaMallocHost((void **)&p, sizeInBytes));
  return p;
}

void CUFFreePinned(void *p) { CUDA_REPORT_IF_ERROR(cudaFreeHost(p)); }

void *CUFAllocDevice(std::size_t sizeInBytes) {
  void *p;
  CUDA_REPORT_IF_ERROR(cudaMalloc(&p, sizeInBytes));
  return p;
}

void CUFFreeDevice(void *p) { CUDA_REPORT_IF_ERROR(cudaFree(p)); }

void *CUFAllocManaged(std::size_t sizeInBytes) {
  void *p;
  CUDA_REPORT_IF_ERROR(
      cudaMallocManaged((void **)&p, sizeInBytes, cudaMemAttachGlobal));
  return reinterpret_cast<void *>(p);
}

void CUFFreeManaged(void *p) { CUDA_REPORT_IF_ERROR(cudaFree(p)); }

void *CUFAllocUnified(std::size_t sizeInBytes) {
  // Call alloc managed for the time being.
  return CUFAllocManaged(sizeInBytes);
}

void CUFFreeUnified(void *p) {
  // Call free managed for the time being.
  CUFFreeManaged(p);
}

} // namespace Fortran::runtime::cuda
