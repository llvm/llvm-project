//===-- include/flang/Runtime/CUDA/allocator.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_ALLOCATOR_H_
#define FORTRAN_RUNTIME_CUDA_ALLOCATOR_H_

#include "common.h"
#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime::cuda {

extern "C" {

void RTDECL(CUFRegisterAllocator)();
}

void *CUFAllocPinned(std::size_t, std::int64_t = kCudaNoStream);
void CUFFreePinned(void *);

void *CUFAllocDevice(std::size_t, std::int64_t);
void CUFFreeDevice(void *);

void *CUFAllocManaged(std::size_t, std::int64_t = kCudaNoStream);
void CUFFreeManaged(void *);

void *CUFAllocUnified(std::size_t, std::int64_t = kCudaNoStream);
void CUFFreeUnified(void *);

} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_ALLOCATOR_H_
