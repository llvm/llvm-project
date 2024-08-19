//===-- include/flang/Runtime/CUDA/descriptor.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_DESCRIPTOR_H_
#define FORTRAN_RUNTIME_CUDA_DESCRIPTOR_H_

#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/entry-names.h"
#include <cstddef>

namespace Fortran::runtime::cuda {

extern "C" {

// Allocate a descriptor in managed.
Descriptor *RTDECL(CUFAllocDesciptor)(
    std::size_t, const char *sourceFile = nullptr, int sourceLine = 0);

// Deallocate a descriptor allocated in managed or unified memory.
void RTDECL(CUFFreeDesciptor)(
    Descriptor *, const char *sourceFile = nullptr, int sourceLine = 0);

} // extern "C"
} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_DESCRIPTOR_H_
