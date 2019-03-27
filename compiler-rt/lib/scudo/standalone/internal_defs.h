//===-- internal_defs.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_INTERNAL_DEFS_H_
#define SCUDO_INTERNAL_DEFS_H_

#include "platform.h"

#include <stdint.h>

#ifndef SCUDO_DEBUG
#define SCUDO_DEBUG 0
#endif

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

// String related macros.

#define STRINGIFY_(S) #S
#define STRINGIFY(S) STRINGIFY_(S)
#define CONCATENATE_(S, C) S##C
#define CONCATENATE(S, C) CONCATENATE_(S, C)

// Attributes & builtins related macros.

#define INTERFACE __attribute__((visibility("default")))
#define WEAK __attribute__((weak))
#define INLINE inline
#define ALWAYS_INLINE inline __attribute__((always_inline))
#define ALIAS(x) __attribute__((alias(x)))
// Please only use the ALIGNED macro before the type. Using ALIGNED after the
// variable declaration is not portable.
#define ALIGNED(x) __attribute__((aligned(x)))
#define FORMAT(f, a) __attribute__((format(printf, f, a)))
#define NOINLINE __attribute__((noinline))
#define NORETURN __attribute__((noreturn))
#define THREADLOCAL __thread
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#if defined(__i386__) || defined(__x86_64__)
// __builtin_prefetch(x) generates prefetchnt0 on x86
#define PREFETCH(x) __asm__("prefetchnta (%0)" : : "r"(x))
#else
#define PREFETCH(x) __builtin_prefetch(x)
#endif
#define UNUSED __attribute__((unused))
#define USED __attribute__((used))
#define NOEXCEPT noexcept

namespace scudo {

typedef unsigned long uptr;
typedef signed long sptr;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;

// The following two functions have platform specific implementations.
void outputRaw(const char *Buffer);
void NORETURN die();

#define RAW_CHECK_MSG(Expr, Msg)                                               \
  do {                                                                         \
    if (UNLIKELY(!(Expr))) {                                                   \
      outputRaw(Msg);                                                          \
      die();                                                                   \
    }                                                                          \
  } while (false)

#define RAW_CHECK(Expr) RAW_CHECK_MSG(Expr, #Expr)

// TODO(kostyak): use reportCheckFailed when checked-in.
#define CHECK_IMPL(c1, op, c2)                                                 \
  do {                                                                         \
    u64 v1 = (u64)(c1);                                                        \
    u64 v2 = (u64)(c2);                                                        \
    if (UNLIKELY(!(v1 op v2))) {                                               \
      outputRaw("CHECK failed: (" #c1 ") " #op " (" #c2 ")\n");                \
      die();                                                                   \
    }                                                                          \
  } while (false)

#define CHECK(a) CHECK_IMPL((a), !=, 0)
#define CHECK_EQ(a, b) CHECK_IMPL((a), ==, (b))
#define CHECK_NE(a, b) CHECK_IMPL((a), !=, (b))
#define CHECK_LT(a, b) CHECK_IMPL((a), <, (b))
#define CHECK_LE(a, b) CHECK_IMPL((a), <=, (b))
#define CHECK_GT(a, b) CHECK_IMPL((a), >, (b))
#define CHECK_GE(a, b) CHECK_IMPL((a), >=, (b))

#if SCUDO_DEBUG
#define DCHECK(a) CHECK(a)
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#else
#define DCHECK(a)
#define DCHECK_EQ(a, b)
#define DCHECK_NE(a, b)
#define DCHECK_LT(a, b)
#define DCHECK_LE(a, b)
#define DCHECK_GT(a, b)
#define DCHECK_GE(a, b)
#endif

// The superfluous die() call effectively makes this macro NORETURN.
#define UNREACHABLE(msg)                                                       \
  do {                                                                         \
    CHECK(0 && msg);                                                           \
    die();                                                                     \
  } while (0)

#define COMPILER_CHECK(Pred) static_assert(Pred, "")

enum LinkerInitialized { LINKER_INITIALIZED = 0 };

} // namespace scudo

#endif // SCUDO_INTERNAL_DEFS_H_
