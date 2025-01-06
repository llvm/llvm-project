//===-- include/flang/Runtime/CUDA/descriptor.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_DESCRIPTOR_H_
#define FORTRAN_RUNTIME_CUDA_DESCRIPTOR_H_

#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"
#include <cstddef>

namespace Fortran::runtime::cuda {

extern "C" {

/// Allocate a descriptor in managed.
Descriptor *RTDECL(CUFAllocDescriptor)(
    std::size_t, const char *sourceFile = nullptr, int sourceLine = 0);

/// Deallocate a descriptor allocated in managed or unified memory.
void RTDECL(CUFFreeDescriptor)(
    Descriptor *, const char *sourceFile = nullptr, int sourceLine = 0);

/// Retrieve the device pointer from the host one.
void *RTDECL(CUFGetDeviceAddress)(
    void *hostPtr, const char *sourceFile = nullptr, int sourceLine = 0);

/// Sync the \p src descriptor to the \p dst descriptor.
void RTDECL(CUFDescriptorSync)(Descriptor *dst, const Descriptor *src,
    const char *sourceFile = nullptr, int sourceLine = 0);

/// Get the device address of registered with the \p hostPtr and sync them.
void RTDECL(CUFSyncGlobalDescriptor)(
    void *hostPtr, const char *sourceFile = nullptr, int sourceLine = 0);

} // extern "C"

} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_DESCRIPTOR_H_
