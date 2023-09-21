//===-- Freestanding version of bit_cast  -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_BIT_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_BIT_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h" // LIBC_HAS_BUILTIN

namespace __llvm_libc::cpp {

#if LIBC_HAS_BUILTIN(__builtin_bit_cast)
#define LLVM_LIBC_HAS_BUILTIN_BIT_CAST
#endif

#if LIBC_HAS_BUILTIN(__builtin_memcpy_inline)
#define LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE
#endif

// This function guarantees the bitcast to be optimized away by the compiler for
// GCC >= 8 and Clang >= 6.
template <class To, class From>
LIBC_INLINE constexpr To bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From), "To and From must be of same size");
  static_assert(cpp::is_trivially_copyable<To>::value &&
                    cpp::is_trivially_copyable<From>::value,
                "Cannot bit-cast instances of non-trivially copyable classes.");
#if defined(LLVM_LIBC_HAS_BUILTIN_BIT_CAST)
  return __builtin_bit_cast(To, from);
#else
  static_assert(cpp::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");
  To to;
  char *dst = reinterpret_cast<char *>(&to);
  const char *src = reinterpret_cast<const char *>(&from);
#if defined(LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE)
  __builtin_memcpy_inline(dst, src, sizeof(To));
#else
  for (unsigned i = 0; i < sizeof(To); ++i)
    dst[i] = src[i];
#endif // defined(LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE)
  return to;
#endif // defined(LLVM_LIBC_HAS_BUILTIN_BIT_CAST)
}

template <class To, class From>
LIBC_INLINE constexpr To bit_or_static_cast(const From &from) {
  if constexpr (sizeof(To) == sizeof(From)) {
    return bit_cast<To>(from);
  } else {
    return static_cast<To>(from);
  }
}

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_BIT_H
