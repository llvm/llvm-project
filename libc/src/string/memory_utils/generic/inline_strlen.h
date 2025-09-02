//===-- Strlen for generic SIMD types -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_INLINE_STRLEN_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_INLINE_STRLEN_H

#include "src/__support/CPP/simd.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Exploit the underlying integer representation to do a variable shift.
LIBC_INLINE constexpr cpp::simd_mask<char> shift_mask(cpp::simd_mask<char> m,
                                                      size_t shift) {
  using bitmask_ty = cpp::internal::get_as_integer_type_t<cpp::simd_mask<char>>;
  bitmask_ty r = cpp::bit_cast<bitmask_ty>(m) >> shift;
  return cpp::bit_cast<cpp::simd_mask<char>>(r);
}

[[clang::no_sanitize("address")]] LIBC_INLINE size_t
string_length(const char *src) {
  constexpr cpp::simd<char> null_byte = cpp::splat('\0');

  size_t alignment = alignof(cpp::simd<char>);
  const cpp::simd<char> *aligned = reinterpret_cast<const cpp::simd<char> *>(
      __builtin_align_down(src, alignment));

  cpp::simd<char> chars = cpp::load_aligned<cpp::simd<char>>(aligned);
  cpp::simd_mask<char> mask = cpp::simd_cast<bool>(chars == null_byte);
  size_t offset = src - reinterpret_cast<const char *>(aligned);
  if (cpp::any_of(shift_mask(mask, offset)))
    return cpp::find_first_set(shift_mask(mask, offset));

  for (;;) {
    cpp::simd<char> chars = cpp::load_aligned<cpp::simd<char>>(++aligned);
    cpp::simd_mask<char> mask = cpp::simd_cast<bool>(chars == null_byte);
    if (cpp::any_of(mask))
      return (reinterpret_cast<const char *>(aligned) - src) +
             cpp::find_first_set(mask);
  }
}
} // namespace internal

namespace string_length_impl = internal;
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_INLINE_STRLEN_H
