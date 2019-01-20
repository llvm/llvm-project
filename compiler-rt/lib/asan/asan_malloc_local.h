//===-- asan_malloc_local.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Provide interfaces to check for and handle local pool memory allocation.
//===----------------------------------------------------------------------===//

#ifndef ASAN_MALLOC_LOCAL_H
#define ASAN_MALLOC_LOCAL_H

#include "sanitizer_common/sanitizer_platform.h"
#include "asan_internal.h"

// On RTEMS, we use the local pool to handle memory allocation when the ASan
// run-time is not up.
static INLINE bool EarlyMalloc() {
  return SANITIZER_RTEMS && (!__asan::asan_inited ||
                             __asan::asan_init_is_running);
}

void* MemalignFromLocalPool(uptr alignment, uptr size);

#if SANITIZER_RTEMS

bool IsFromLocalPool(const void *ptr);

#define ALLOCATE_FROM_LOCAL_POOL UNLIKELY(EarlyMalloc())
#define IS_FROM_LOCAL_POOL(ptr) UNLIKELY(IsFromLocalPool(ptr))

#else  // SANITIZER_RTEMS

#define ALLOCATE_FROM_LOCAL_POOL 0
#define IS_FROM_LOCAL_POOL(ptr) 0

#endif  // SANITIZER_RTEMS

#endif  // ASAN_MALLOC_LOCAL_H
