//===-- Implementation of bcmp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BCMP_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BCMP_IMPLEMENTATIONS_H

#include "src/__support/architectures.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/elements.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

// Fixed-size difference between 'lhs' and 'rhs'.
template <typename Element> bool differs(const char *lhs, const char *rhs) {
  return !Element::equals(lhs, rhs);
}
// Runtime-size difference between 'lhs' and 'rhs'.
template <typename Element>
bool differs(const char *lhs, const char *rhs, size_t size) {
  return !Element::equals(lhs, rhs, size);
}

static inline int inline_bcmp(const char *lhs, const char *rhs, size_t count) {
#if defined(LLVM_LIBC_ARCH_X86)
  using namespace ::__llvm_libc::x86;
#elif defined(LLVM_LIBC_ARCH_AARCH64)
  using namespace ::__llvm_libc::aarch64;
#else
  using namespace ::__llvm_libc::scalar;
#endif
  if (count == 0)
    return 0;
  if (count == 1)
    return differs<_1>(lhs, rhs);
  if (count == 2)
    return differs<_2>(lhs, rhs);
  if (count == 3)
    return differs<_3>(lhs, rhs);
  if (count <= 8)
    return differs<HeadTail<_4>>(lhs, rhs, count);
  if (count <= 16)
    return differs<HeadTail<_8>>(lhs, rhs, count);
  if (count <= 32)
    return differs<HeadTail<_16>>(lhs, rhs, count);
  if (count <= 64)
    return differs<HeadTail<_32>>(lhs, rhs, count);
  if (count <= 128)
    return differs<HeadTail<_64>>(lhs, rhs, count);
  return differs<Align<_32>::Then<Loop<_32>>>(lhs, rhs, count);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BCMP_IMPLEMENTATIONS_H
