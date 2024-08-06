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

namespace Fortran::runtime::cuda {

RT_API_ATTRS void CUFRegisterAllocator();

RT_API_ATTRS void *CUFAllocPinned(std::size_t);
RT_API_ATTRS void CUFFreePinned(void *);

RT_API_ATTRS void *CUFAllocDevice(std::size_t);
RT_API_ATTRS void CUFFreeDevice(void *);

RT_API_ATTRS void *CUFAllocManaged(std::size_t);
RT_API_ATTRS void CUFFreeManaged(void *);

RT_API_ATTRS void *CUFAllocUnified(std::size_t);
RT_API_ATTRS void CUFFreeUnified(void *);

} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_ALLOCATOR_H_
