//===-- wrappers_c_checks.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_CHECKS_H_
#define SCUDO_CHECKS_H_

#include "common.h"

#include <errno.h>

namespace scudo {

// A common errno setting logic shared by almost all Scudo C wrappers.
INLINE void *setErrnoOnNull(void *Ptr) {
  if (UNLIKELY(!Ptr))
    errno = ENOMEM;
  return Ptr;
}

// Checks return true on failure.

// Checks aligned_alloc() parameters, verifies that the alignment is a power of
// two and that the size is a multiple of alignment.
INLINE bool checkAlignedAllocAlignmentAndSize(uptr Alignment, uptr Size) {
  return Alignment == 0 || !isPowerOfTwo(Alignment) ||
         !isAligned(Size, Alignment);
}

// Checks posix_memalign() parameters, verifies that alignment is a power of two
// and a multiple of sizeof(void *).
INLINE bool checkPosixMemalignAlignment(uptr Alignment) {
  return Alignment == 0 || !isPowerOfTwo(Alignment) ||
         !isAligned(Alignment, sizeof(void *));
}

// Returns true if calloc(Size, N) overflows on Size*N calculation. The typical
// way would be to check for (UINTPTR_MAX / Size) < N, but the division ends up
// being very costly, so use a builtin supported by recent clang & GCC.
INLINE bool checkForCallocOverflow(uptr Size, uptr N, uptr *Product) {
  return __builtin_umull_overflow(Size, N, Product);
}

// Returns true if the size passed to pvalloc overflows when rounded to the next
// multiple of PageSize.
INLINE bool checkForPvallocOverflow(uptr Size, uptr PageSize) {
  return roundUpTo(Size, PageSize) < Size;
}

} // namespace scudo

#endif // SCUDO_CHECKS_H_
