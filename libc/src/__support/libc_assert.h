//===-- Definition of a libc internal assert macro --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_LIBC_ASSERT_H
#define LLVM_LIBC_SRC___SUPPORT_LIBC_ASSERT_H

#include "src/__support/macros/attributes.h" // For LIBC_INLINE
#include "src/__support/macros/config.h"
#include "src/__support/macros/hardening.h"
#include "src/__support/macros/macro-utils.h"
#include "src/__support/macros/optimization.h" // For LIBC_UNLIKELY
#include "src/__support/macros/properties/os.h"

#ifdef LIBC_FULL_BUILD
#include "src/__support/OSUtil/assert_fail.h"
#endif

//===----------------------------------------------------------------------===//
// LIBC_REQUIRE(COND, MSG) (always-on assert regardless of NDEBUG)
//===----------------------------------------------------------------------===//
#ifndef LIBC_FULL_BUILD
#ifdef LIBC_TARGET_OS_IS_LINUX
// __assert_fail is in Linux Standard Base (LSB), hence we should always be able
// to use it here.
#include <assert.h>
#define LIBC_REQUIRE(COND, MSG)                                                \
  do {                                                                         \
    if (LIBC_UNLIKELY(!(COND)))                                                \
      __assert_fail(MSG, __FILE__, __LINE__, __PRETTY_FUNCTION__);             \
  } while (false)
#else
// Fallback path will just trap: we cannot reliably do anything else.
#define LIBC_REQUIRE(COND, MSG)                                                \
  do {                                                                         \
    if (LIBC_UNLIKELY(!(COND)))                                                \
      __builtin_trap();                                                        \
  } while (false)
#endif // LIBC_TARGET_OS_IS_LINUX
#else
#define LIBC_REQUIRE(COND, MSG)                                                \
  do {                                                                         \
    if (LIBC_UNLIKELY(!(COND)))                                                \
      LIBC_NAMESPACE::internal::assert_fail(MSG, __FILE__, __LINE__,           \
                                            __PRETTY_FUNCTION__);              \
  } while (false)
#endif // LIBC_FULL_BUILD

//===----------------------------------------------------------------------===//
// LIBC_ASSERT(COND) (NDEBUG guarded assertion)
//===----------------------------------------------------------------------===//
#if defined(LIBC_COPT_USE_C_ASSERT) || !defined(LIBC_FULL_BUILD)

// The build is configured to just use the public <assert.h> API
// for libc's internal assertions.
#ifndef LIBC_ASSERT
#include <assert.h>
#define LIBC_ASSERT(COND) assert(COND)
#endif // LIBC_ASSERT

#else // Not LIBC_COPT_USE_C_ASSERT
#ifdef LIBC_ASSERT
#error "Unexpected: LIBC_ASSERT macro already defined"
#endif

#ifdef NDEBUG
#define LIBC_ASSERT(COND)                                                      \
  do {                                                                         \
  } while (false)
#else
// Forward to LIBC_REQUIRE with the condition stringified.
#define LIBC_ASSERT(COND) LIBC_REQUIRE(COND, #COND)
#endif // NDEBUG

#endif // LIBC_COPT_USE_C_ASSERT

//===----------------------------------------------------------------------===//
// Hardening runtime check
//===----------------------------------------------------------------------===//

#if LIBC_COPT_HARDENING_MODE == LIBC_HARDENING_MODE_NONE
#define LIBC_HEAP_INTEGRITY_CHECK(COND, MSG) ((void)0)
#elif LIBC_COPT_HARDENING_MODE == LIBC_HARDENING_MODE_FAST
#define LIBC_HEAP_INTEGRITY_CHECK(COND, MSG) ((void)0)
#elif LIBC_COPT_HARDENING_MODE == LIBC_HARDENING_MODE_EXTENSIVE
#define LIBC_HEAP_INTEGRITY_CHECK(COND, MSG) LIBC_REQUIRE(COND, MSG)
#elif LIBC_COPT_HARDENING_MODE == LIBC_HARDENING_MODE_DEBUG
#define LIBC_HEAP_INTEGRITY_CHECK(COND, MSG) LIBC_REQUIRE(COND, MSG)
#else
#error "Unsupported hardening mode"
#endif
#endif // LLVM_LIBC_SRC___SUPPORT_LIBC_ASSERT_H
