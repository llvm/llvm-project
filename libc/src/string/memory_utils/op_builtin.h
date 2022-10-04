//===-- Implementation using the __builtin_XXX_inline ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides generic C++ building blocks to compose memory functions.
// They rely on the compiler to generate the best possible code through the use
// of the `__builtin_XXX_inline` builtins. These builtins are currently only
// available in Clang.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_BUILTIN_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_BUILTIN_H

#include "src/string/memory_utils/utils.h"

namespace __llvm_libc::builtin {

///////////////////////////////////////////////////////////////////////////////
// Memcpy
template <size_t Size> struct Memcpy {
  static constexpr size_t SIZE = Size;
  static inline void block(Ptr __restrict dst, CPtr __restrict src) {
#ifdef LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE
    return __builtin_memcpy_inline(dst, src, SIZE);
#else
    deferred_static_assert("Missing __builtin_memcpy_inline");
    (void)dst;
    (void)src;
#endif
  }

  static inline void tail(Ptr __restrict dst, CPtr __restrict src,
                          size_t count) {
    block(dst + count - SIZE, src + count - SIZE);
  }

  static inline void head_tail(Ptr __restrict dst, CPtr __restrict src,
                               size_t count) {
    block(dst, src);
    tail(dst, src, count);
  }

  static inline void loop_and_tail(Ptr __restrict dst, CPtr __restrict src,
                                   size_t count) {
    size_t offset = 0;
    do {
      block(dst + offset, src + offset);
      offset += SIZE;
    } while (offset < count - SIZE);
    tail(dst, src, count);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Memset
template <size_t Size> struct Memset {
  using ME = Memset;
  static constexpr size_t SIZE = Size;
  static inline void block(Ptr dst, uint8_t value) {
#ifdef LLVM_LIBC_HAS_BUILTIN_MEMSET_INLINE
    __builtin_memset_inline(dst, value, Size);
#else
    deferred_static_assert("Missing __builtin_memset_inline");
    (void)dst;
    (void)value;
#endif
  }

  static inline void tail(Ptr dst, uint8_t value, size_t count) {
    block(dst + count - SIZE, value);
  }

  static inline void head_tail(Ptr dst, uint8_t value, size_t count) {
    block(dst, value);
    tail(dst, value, count);
  }

  static inline void loop_and_tail(Ptr dst, uint8_t value, size_t count) {
    size_t offset = 0;
    do {
      block(dst + offset, value);
      offset += SIZE;
    } while (offset < count - SIZE);
    tail(dst, value, count);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Bcmp
template <size_t Size> struct Bcmp {
  using ME = Bcmp;
  static constexpr size_t SIZE = Size;
  static inline BcmpReturnType block(CPtr, CPtr) {
    deferred_static_assert("Missing __builtin_memcmp_inline");
    return BcmpReturnType::ZERO();
  }

  static inline BcmpReturnType tail(CPtr, CPtr, size_t) {
    deferred_static_assert("Not implemented");
    return BcmpReturnType::ZERO();
  }

  static inline BcmpReturnType head_tail(CPtr, CPtr, size_t) {
    deferred_static_assert("Not implemented");
    return BcmpReturnType::ZERO();
  }

  static inline BcmpReturnType loop_and_tail(CPtr, CPtr, size_t) {
    deferred_static_assert("Not implemented");
    return BcmpReturnType::ZERO();
  }
};

///////////////////////////////////////////////////////////////////////////////
// Memcmp
template <size_t Size> struct Memcmp {
  using ME = Memcmp;
  static constexpr size_t SIZE = Size;
  static inline MemcmpReturnType block(CPtr, CPtr) {
    deferred_static_assert("Missing __builtin_memcmp_inline");
    return MemcmpReturnType::ZERO();
  }

  static inline MemcmpReturnType tail(CPtr, CPtr, size_t) {
    deferred_static_assert("Not implemented");
    return MemcmpReturnType::ZERO();
  }

  static inline MemcmpReturnType head_tail(CPtr, CPtr, size_t) {
    deferred_static_assert("Not implemented");
    return MemcmpReturnType::ZERO();
  }

  static inline MemcmpReturnType loop_and_tail(CPtr, CPtr, size_t) {
    deferred_static_assert("Not implemented");
    return MemcmpReturnType::ZERO();
  }
};

} // namespace __llvm_libc::builtin

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_BUILTIN_H
