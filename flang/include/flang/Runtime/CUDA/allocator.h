//===-- include/flang/Runtime/CUDA/allocator.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_ALLOCATOR_H_
#define FORTRAN_RUNTIME_CUDA_ALLOCATOR_H_

#include "flang/Runtime/descriptor.h"

static constexpr unsigned kPinnedAllocatorPos = 1;
static constexpr unsigned kDeviceAllocatorPos = 2;
static constexpr unsigned kManagedAllocatorPos = 3;

#define CUDA_REPORT_IF_ERROR(expr) \
  [](CUresult result) { \
    if (!result) \
      return; \
    const char *name = nullptr; \
    cuGetErrorName(result, &name); \
    if (!name) \
      name = "<unknown>"; \
    Terminator terminator{__FILE__, __LINE__}; \
    terminator.Crash("'%s' failed with '%s'", #expr, name); \
  }(expr)

namespace Fortran::runtime::cuf {

void CUFRegisterAllocator();

void *CUFAllocPinned(std::size_t);
void CUFFreePinned(void *);

void *CUFAllocDevice(std::size_t);
void CUFFreeDevice(void *);

void *CUFAllocManaged(std::size_t);
void CUFFreeManaged(void *);

} // namespace Fortran::runtime::cuf
#endif // FORTRAN_RUNTIME_CUDA_ALLOCATOR_H_
