//===-- runtime/CUDA/allocator.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/allocator.h"
#include "../allocator-registry.h"
#include "../derived.h"
#include "../stat.h"
#include "../terminator.h"
#include "../type-info.h"
#include "flang/Common/Fortran.h"
#include "flang/ISO_Fortran_binding_wrapper.h"

#include "cuda.h"

namespace Fortran::runtime::cuf {

void CUFRegisterAllocator() {
  allocatorRegistry.Register(
      kPinnedAllocatorPos, {&CUFAllocPinned, CUFFreePinned});
  allocatorRegistry.Register(
      kDeviceAllocatorPos, {&CUFAllocDevice, CUFFreeDevice});
  allocatorRegistry.Register(
      kManagedAllocatorPos, {&CUFAllocManaged, CUFFreeManaged});
}

void *CUFAllocPinned(std::size_t sizeInBytes) {
  void *p;
  CUDA_REPORT_IF_ERROR(cuMemAllocHost(&p, sizeInBytes));
  return p;
}

void CUFFreePinned(void *p) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(p)));
}

void *CUFAllocDevice(std::size_t sizeInBytes) {
  CUdeviceptr p = 0;
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&p, sizeInBytes));
  return reinterpret_cast<void *>(p);
}

void CUFFreeDevice(void *p) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(p)));
}

void *CUFAllocManaged(std::size_t sizeInBytes) {
  CUdeviceptr p = 0;
  CUDA_REPORT_IF_ERROR(
      cuMemAllocManaged(&p, sizeInBytes, CU_MEM_ATTACH_GLOBAL));
  return reinterpret_cast<void *>(p);
}

void CUFFreeManaged(void *p) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(p)));
}

} // namespace Fortran::runtime::cuf
