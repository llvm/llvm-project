//===-- Utilities for getting secure randomness -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPRNG_H
#define LLVM_LIBC_SRC___SUPPORT_CPRNG_H

#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"

#define __need_size_t
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace cprng {
size_t fill_buffer(char *buffer, size_t length);

template <typename T> LIBC_INLINE cpp::optional<T> generate() {
  static_assert(cpp::is_arithmetic_v<T>, "T must be an arithmetic type.");
  union {
    T result;
    char bytes[sizeof(T)];
  };
  if (fill_buffer(bytes, sizeof(bytes)) == sizeof(bytes))
    return result;
  return cpp::nullopt;
}

// based on https://jacquesheunis.com/post/bounded-random/
LIBC_INLINE cpp::optional<uint32_t> generate_bounded_u32(uint32_t bound) {
  auto lifted_bound = static_cast<uint64_t>(bound);
  cpp::optional<uint64_t> seed = generate<uint64_t>();
  if (!seed)
    return cpp::nullopt;
  uint64_t r0 = *seed & 0xFFFFFFFFu;
  uint64_t r1 = (*seed >> 32) & 0xFFFFFFFFu;
  uint64_t prod0 = static_cast<uint64_t>(r0) * lifted_bound;
  uint64_t prod0_hi = (prod0 >> 32) & 0xFFFFFFFFu;
  uint64_t prod0_lo = prod0 & 0xFFFFFFFFu;
  uint64_t prod1 = static_cast<uint64_t>(r1) * lifted_bound;
  uint64_t prod1_hi = (prod1 >> 32) & 0xFFFFFFFFu;
  uint64_t sum = prod0_lo + prod1_hi;
  uint64_t sum_hi = (sum >> 32) & 0xFFFFFFFFu;
  return static_cast<uint32_t>(prod0_hi + sum_hi);
}

} // namespace cprng
} // namespace LIBC_NAMESPACE_DECL
#endif // LLVM_LIBC_SRC___SUPPORT_CPRNG_H
