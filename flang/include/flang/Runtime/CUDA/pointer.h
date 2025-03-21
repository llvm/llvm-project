//===-- include/flang/Runtime/CUDA/pointer.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_POINTER_H_
#define FORTRAN_RUNTIME_CUDA_POINTER_H_

#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime::cuda {

extern "C" {

/// Perform allocation of the descriptor.
int RTDECL(CUFPointerAllocate)(Descriptor &, int64_t stream = -1,
    bool *pinned = nullptr, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

/// Perform allocation of the descriptor with synchronization of it when
/// necessary.
int RTDECL(CUFPointerAllocateSync)(Descriptor &, int64_t stream = -1,
    bool *pinned = nullptr, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

/// Perform allocation of the descriptor without synchronization. Assign data
/// from source.
int RTDEF(CUFPointerAllocateSource)(Descriptor &pointer,
    const Descriptor &source, int64_t stream = -1, bool *pinned = nullptr,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

/// Perform allocation of the descriptor with synchronization of it when
/// necessary. Assign data from source.
int RTDEF(CUFPointerAllocateSourceSync)(Descriptor &pointer,
    const Descriptor &source, int64_t stream = -1, bool *pinned = nullptr,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

} // extern "C"

} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_POINTER_H_
