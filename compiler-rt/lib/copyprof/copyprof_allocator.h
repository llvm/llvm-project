//===-- copyprof_allocator.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the standalone memory allocator interface for CopyProf,
/// enabling isolated memory management for internal runtime structures.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_ALLOCATOR_H
#define COPYPROF_ALLOCATOR_H

#include "copyprof_internal.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __copyprof {

// Minimum alignment (in bytes) of allocations returned by the CopyProf
// allocator.
constexpr uptr kMinAlignment = 8;

// Initializes the copyprof allocator at startup.
void InitializeCopyProfAllocator();

// CopyProf specific memory allocation / deallocation routines and libc
// surrogates.
void* CopyProfAllocate(uptr size, uptr align);
void CopyProfDeallocate(void* ptr, uptr size, uptr align);
void* CopyProfMalloc(uptr size);
void CopyProfFree(void* ptr);
void* CopyProfCalloc(uptr num, uptr size);
void* CopyProfRealloc(void* ptr, uptr size);
void* CopyProfReallocarray(void* ptr, uptr num, uptr size);
void* CopyProfValloc(uptr size);
void* CopyProfPvalloc(uptr size);
int CopyProfPosixMemalign(void** memptr, uptr alignment, uptr size);
uptr CopyProfMallocUsableSize(const void* ptr);

}  // namespace __copyprof

#endif  // COPYPROF_ALLOCATOR_H
