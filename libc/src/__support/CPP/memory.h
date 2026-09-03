//===-- Analogous to <memory> ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_MEMORY_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_MEMORY_H

#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

template <class T> LIBC_INLINE T *start_lifetime_as(void *ptr) noexcept {
#if __has_builtin(__builtin_is_implicit_lifetime)
  static_assert(__builtin_is_implicit_lifetime(T),
                "T must be an implicitly lifetime type");
#endif
#if __has_builtin(__builtin_start_lifetime_as)
  // MSVC STL is using this builtin, upstream clang does not have it yet as of
  // April, 2026
  return __builtin_start_lifetime_as<T>(ptr);
#elif defined(__GNUC__)
  // Mark the ptr as opaque value to temporarily serve as a TBAA barrier.
  // However, this does not work with tysan because tysan stores metadata
  // associated with the address and does not take account of this kind of
  // assembly trick. See
  // https://github.com/llvm/llvm-project/issues/193248.
  // We don't have other good strategy yet.
  auto *res = static_cast<T *>(ptr);
  asm volatile("" : "=g"(res), "=m"(*res) : "0"(res), "m"(*res));
  return res;
#else
  return static_cast<T *>(ptr);
#endif
}

// forward other CV-qualified variants
template <class T>
LIBC_INLINE const T *start_lifetime_as(const void *ptr) noexcept {
  return start_lifetime_as<T>(const_cast<void *>(ptr));
}

template <class T>
LIBC_INLINE volatile T *start_lifetime_as(volatile void *ptr) noexcept {
  return start_lifetime_as<T>(const_cast<void *>(ptr));
}

template <class T>
LIBC_INLINE const volatile T *
start_lifetime_as(const volatile void *ptr) noexcept {
  return start_lifetime_as<T>(const_cast<void *>(ptr));
}

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_MEMORY_H
