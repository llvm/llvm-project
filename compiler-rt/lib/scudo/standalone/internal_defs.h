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

// Various integer constants.

#undef __INT64_C
#undef __UINT64_C
#undef UINTPTR_MAX
#if SCUDO_WORDSIZE == 64U
#define __INT64_C(c) c##L
#define __UINT64_C(c) c##UL
#define UINTPTR_MAX (18446744073709551615UL)
#else
#define __INT64_C(c) c##LL
#define __UINT64_C(c) c##ULL
#define UINTPTR_MAX (4294967295U)
#endif // SCUDO_WORDSIZE == 64U
#undef INT32_MIN
#define INT32_MIN (-2147483647 - 1)
#undef INT32_MAX
#define INT32_MAX (2147483647)
#undef UINT32_MAX
#define UINT32_MAX (4294967295U)
#undef INT64_MIN
#define INT64_MIN (-__INT64_C(9223372036854775807) - 1)
#undef INT64_MAX
#define INT64_MAX (__INT64_C(9223372036854775807))
#undef UINT64_MAX
#define UINT64_MAX (__UINT64_C(18446744073709551615))

enum LinkerInitialized { LINKER_INITIALIZED = 0 };

// Various CHECK related macros.

#define COMPILER_CHECK(Pred) static_assert(Pred, "")

// TODO(kostyak): implement at a later check-in.
#define CHECK_IMPL(c1, op, c2)                                                 \
  do {                                                                         \
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

// TODO(kostyak): implement at a later check-in.
#define UNREACHABLE(msg)                                                       \
  do {                                                                         \
  } while (false)

} // namespace scudo

#endif // SCUDO_INTERNAL_DEFS_H_
