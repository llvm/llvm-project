//===-- Convenient sanitizer macros -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_SANITIZER_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_SANITIZER_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h" //LIBC_HAS_FEATURE

//-----------------------------------------------------------------------------
// Functions to unpoison memory
//-----------------------------------------------------------------------------

#if LIBC_HAS_FEATURE(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define LIBC_HAS_ADDRESS_SANITIZER
#endif

#if LIBC_HAS_FEATURE(memory_sanitizer)
#define LIBC_HAS_MEMORY_SANITIZER
#endif

#ifdef LIBC_HAS_MEMORY_SANITIZER
// Only perform MSAN unpoison in non-constexpr context.
#include <sanitizer/msan_interface.h>
#define LIBC_MSAN_UNPOISON(addr, size)                                         \
  do {                                                                         \
    if (!__builtin_is_constant_evaluated())                                    \
      __msan_unpoison(addr, size);                                             \
  } while (0)
#else
#define LIBC_MSAN_UNPOISON(ptr, size)
#endif

#ifdef LIBC_HAS_ADDRESS_SANITIZER
#include <sanitizer/asan_interface.h>
#define LIBC_ASAN_POISON_MEMORY_REGION(addr, size)                             \
  __asan_poison_memory_region((addr), (size))
#define LIBC_ASAN_UNPOISON_MEMORY_REGION(addr, size)                           \
  __asan_unpoison_memory_region((addr), (size))
#else
#define LIBC_ASAN_POISON_MEMORY_REGION(addr, size) ((void)(addr), (void)(size))
#define LIBC_ASAN_UNPOISON_MEMORY_REGION(addr, size)                           \
  ((void)(addr), (void)(size))
#endif

//-----------------------------------------------------------------------------
// LeakSanitizer utilities
//-----------------------------------------------------------------------------

namespace LIBC_NAMESPACE_DECL {
namespace sanitizer {

extern "C" [[gnu::weak]] void __lsan_ignore_object(const void *);

#if LIBC_HAS_FEATURE(leak_sanitizer) || defined(LIBC_HAS_ADDRESS_SANITIZER)
constexpr bool HAS_LEAK_SANITIZER = true;
#else
constexpr bool HAS_LEAK_SANITIZER = false;
#endif

template <bool Enabled = HAS_LEAK_SANITIZER>
LIBC_INLINE void ignore_object(const void *addr) {
  if constexpr (Enabled) {
    if (&__lsan_ignore_object && addr)
      __lsan_ignore_object(addr);
  } else {
    (void)addr;
  }
}

} // namespace sanitizer
} // namespace LIBC_NAMESPACE_DECL

#define LIBC_LSAN_IGNORE_OBJECT(addr)                                          \
  LIBC_NAMESPACE::sanitizer::ignore_object(addr)

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_SANITIZER_H
