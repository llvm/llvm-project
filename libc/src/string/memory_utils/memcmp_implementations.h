//===-- Implementation of memcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCMP_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCMP_IMPLEMENTATIONS_H

#include "src/__support/architectures.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/elements.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

static inline int inline_memcmp(const char *lhs, const char *rhs,
                                size_t count) {
#if defined(LLVM_LIBC_ARCH_X86)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_X86
  /////////////////////////////////////////////////////////////////////////////
  using namespace __llvm_libc::x86;
  if (count == 0)
    return 0;
  if (count == 1)
    return ThreeWayCompare<_1>(lhs, rhs);
  if (count == 2)
    return ThreeWayCompare<_2>(lhs, rhs);
  if (count == 3)
    return ThreeWayCompare<_3>(lhs, rhs);
  if (count <= 8)
    return ThreeWayCompare<HeadTail<_4>>(lhs, rhs, count);
  if (count <= 16)
    return ThreeWayCompare<HeadTail<_8>>(lhs, rhs, count);
  if (count <= 32)
    return ThreeWayCompare<HeadTail<_16>>(lhs, rhs, count);
  if (count <= 64)
    return ThreeWayCompare<HeadTail<_32>>(lhs, rhs, count);
  if (count <= 128)
    return ThreeWayCompare<HeadTail<_64>>(lhs, rhs, count);
  return ThreeWayCompare<Align<_32>::Then<Loop<_32>>>(lhs, rhs, count);
#elif defined(LLVM_LIBC_ARCH_AARCH64)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_AARCH64
  /////////////////////////////////////////////////////////////////////////////
  using namespace ::__llvm_libc::aarch64;
  if (count == 0) // [0, 0]
    return 0;
  if (count == 1) // [1, 1]
    return ThreeWayCompare<_1>(lhs, rhs);
  if (count == 2) // [2, 2]
    return ThreeWayCompare<_2>(lhs, rhs);
  if (count == 3) // [3, 3]
    return ThreeWayCompare<_3>(lhs, rhs);
  if (count < 8) // [4, 7]
    return ThreeWayCompare<HeadTail<_4>>(lhs, rhs, count);
  if (count < 16) // [8, 15]
    return ThreeWayCompare<HeadTail<_8>>(lhs, rhs, count);
  if (unlikely(count >= 128)) // [128, ∞]
    return ThreeWayCompare<Align<_16>::Then<Loop<_32>>>(lhs, rhs, count);
  if (!Equals<_16>(lhs, rhs)) // [16, 16]
    return ThreeWayCompare<_16>(lhs, rhs);
  if (count < 32) // [17, 31]
    return ThreeWayCompare<Tail<_16>>(lhs, rhs, count);
  if (!Equals<Skip<16>::Then<_16>>(lhs, rhs)) // [32, 32]
    return ThreeWayCompare<Skip<16>::Then<_16>>(lhs, rhs);
  if (count < 64) // [33, 63]
    return ThreeWayCompare<Tail<_32>>(lhs, rhs, count);
  // [64, 127]
  return ThreeWayCompare<Skip<32>::Then<Loop<_16>>>(lhs, rhs, count);
#else
  /////////////////////////////////////////////////////////////////////////////
  // Default
  /////////////////////////////////////////////////////////////////////////////
  using namespace ::__llvm_libc::scalar;

  if (count == 0)
    return 0;
  if (count == 1)
    return ThreeWayCompare<_1>(lhs, rhs);
  if (count == 2)
    return ThreeWayCompare<_2>(lhs, rhs);
  if (count == 3)
    return ThreeWayCompare<_3>(lhs, rhs);
  if (count <= 8)
    return ThreeWayCompare<HeadTail<_4>>(lhs, rhs, count);
  if (count <= 16)
    return ThreeWayCompare<HeadTail<_8>>(lhs, rhs, count);
  if (count <= 32)
    return ThreeWayCompare<HeadTail<_16>>(lhs, rhs, count);
  if (count <= 64)
    return ThreeWayCompare<HeadTail<_32>>(lhs, rhs, count);
  if (count <= 128)
    return ThreeWayCompare<HeadTail<_64>>(lhs, rhs, count);
  return ThreeWayCompare<Align<_32>::Then<Loop<_32>>>(lhs, rhs, count);
#endif
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCMP_IMPLEMENTATIONS_H
