//===-- Shared Scudo Allocator State ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_SCUDO_SCUDO_ALLOCATOR_H
#define LLVM_LIBC_SRC_STDLIB_SCUDO_SCUDO_ALLOCATOR_H

#include "src/__support/macros/config.h"
#include "allocator_config.h"
#include "internal_defs.h"
#include "scudo/interface.h"
#include "wrappers_c_checks.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

inline constexpr size_t SCUDO_MALLOC_ALIGNMENT = FIRST_32_SECOND_64(8U, 16U);

void malloc_postinit();
extern scudo::Allocator<scudo::Config, malloc_postinit> Allocator;

LIBC_INLINE void reportAllocation(void *Ptr, size_t Size) {
  if (SCUDO_ENABLE_HOOKS)
    if (__scudo_allocate_hook && Ptr)
      __scudo_allocate_hook(Ptr, Size);
}

LIBC_INLINE void reportDeallocation(void *Ptr) {
  if (SCUDO_ENABLE_HOOKS)
    if (__scudo_deallocate_hook)
      __scudo_deallocate_hook(Ptr);
}

LIBC_INLINE void reportReallocAllocation(void *OldPtr, void *NewPtr,
                                         size_t Size) {
  DCHECK_NE(NewPtr, nullptr);

  if (SCUDO_ENABLE_HOOKS) {
    if (__scudo_realloc_allocate_hook)
      __scudo_realloc_allocate_hook(OldPtr, NewPtr, Size);
    else if (__scudo_allocate_hook)
      __scudo_allocate_hook(NewPtr, Size);
  }
}

LIBC_INLINE void reportReallocDeallocation(void *OldPtr) {
  if (SCUDO_ENABLE_HOOKS) {
    if (__scudo_realloc_deallocate_hook)
      __scudo_realloc_deallocate_hook(OldPtr);
    else if (__scudo_deallocate_hook)
      __scudo_deallocate_hook(OldPtr);
  }
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_SCUDO_SCUDO_ALLOCATOR_H
