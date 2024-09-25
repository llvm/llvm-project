//===-- include/flang/Runtime/CUDA/memory.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_MEMORY_H_
#define FORTRAN_RUNTIME_CUDA_MEMORY_H_

#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/entry-names.h"
#include <cstddef>

namespace Fortran::runtime::cuda {

extern "C" {

/// Allocate memory on the device.
void *RTDECL(CUFMemAlloc)(std::size_t bytes, unsigned type,
    const char *sourceFile = nullptr, int sourceLine = 0);

/// Free memory allocated on the device.
void RTDECL(CUFMemFree)(void *devicePtr, unsigned type,
    const char *sourceFile = nullptr, int sourceLine = 0);

/// Set value to the data hold by a descriptor. The \p value pointer must be
/// addressable to the same amount of bytes specified by the element size of
/// the descriptor \p desc.
void RTDECL(CUFMemsetDescriptor)(const Descriptor &desc, void *value,
    const char *sourceFile = nullptr, int sourceLine = 0);

/// Data transfer from a pointer to a pointer.
void RTDECL(CUFDataTransferPtrPtr)(void *dst, void *src, std::size_t bytes,
    unsigned mode, const char *sourceFile = nullptr, int sourceLine = 0);

/// Data transfer from a pointer to a descriptor.
void RTDECL(CUFDataTransferDescPtr)(const Descriptor &dst, void *src,
    std::size_t bytes, unsigned mode, const char *sourceFile = nullptr,
    int sourceLine = 0);

/// Data transfer from a descriptor to a pointer.
void RTDECL(CUFDataTransferPtrDesc)(void *dst, const Descriptor &src,
    std::size_t bytes, unsigned mode, const char *sourceFile = nullptr,
    int sourceLine = 0);

/// Data transfer from a descriptor to a descriptor.
void RTDECL(CUFDataTransferDescDesc)(const Descriptor &dst,
    const Descriptor &src, unsigned mode, const char *sourceFile = nullptr,
    int sourceLine = 0);

} // extern "C"
} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_MEMORY_H_
