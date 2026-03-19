//===-- include/flang/Runtime/CUDA/memory.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_MEMORY_H_
#define FORTRAN_RUNTIME_CUDA_MEMORY_H_

#include "flang/Runtime/descriptor-consts.h"
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
void RTDECL(CUFMemsetDescriptor)(Descriptor *desc, void *value,
    const char *sourceFile = nullptr, int sourceLine = 0);

/// Data transfer from a pointer to a pointer.
void RTDECL(CUFDataTransferPtrPtr)(void *dst, void *src, std::size_t bytes,
    unsigned mode, const char *sourceFile = nullptr, int sourceLine = 0);

/// Data transfer from a descriptor to a pointer.
void RTDECL(CUFDataTransferPtrDesc)(void *dst, Descriptor *src,
    std::size_t bytes, unsigned mode, const char *sourceFile = nullptr,
    int sourceLine = 0);

/// Data transfer from a descriptor to a descriptor.
void RTDECL(CUFDataTransferDescDesc)(Descriptor *dst, Descriptor *src,
    unsigned mode, const char *sourceFile = nullptr, int sourceLine = 0);

/// Data transfer from a scalar descriptor to a descriptor.
void RTDECL(CUFDataTransferCstDesc)(Descriptor *dst, Descriptor *src,
    unsigned mode, const char *sourceFile = nullptr, int sourceLine = 0);

/// Data transfer from a descriptor to a descriptor.
void RTDECL(CUFDataTransferDescDescNoRealloc)(Descriptor *dst, Descriptor *src,
    unsigned mode, const char *sourceFile = nullptr, int sourceLine = 0);

/// Data transfer from a descriptor to a global descriptor.
void RTDECL(CUFDataTransferGlobalDescDesc)(Descriptor *dst, Descriptor *src,
    unsigned mode, const char *sourceFile = nullptr, int sourceLine = 0);

} // extern "C"
} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_MEMORY_H_
