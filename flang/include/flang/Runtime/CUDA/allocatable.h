//===-- include/flang/Runtime/CUDA/allocatable.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_ALLOCATABLE_H_
#define FORTRAN_RUNTIME_CUDA_ALLOCATABLE_H_

#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime::cuda {

extern "C" {

/// Perform allocation of the descriptor with synchronization of it when
/// necessary.
int RTDECL(CUFAllocatableAllocate)(Descriptor &, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

/// Perform deallocation of the descriptor with synchronization of it when
/// necessary.
int RTDECL(CUFAllocatableDeallocate)(Descriptor &, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

} // extern "C"

} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_ALLOCATABLE_H_
