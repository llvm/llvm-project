//===-- Definition of a libc internal assert macro --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_LIBC_ASSERT_H
#define LLVM_LIBC_SRC___SUPPORT_LIBC_ASSERT_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/OSUtil/linux/io.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/attributes.h" // For LIBC_INLINE
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LIBC_INLINE void report_assertion_failure(cpp::string_view assertion,
                                          cpp::string_view filename,
                                          cpp::string_view line,
                                          cpp::string_view funcname) {
  write_all_to_stderr({filename, ":", line, ": Assertion failed: '", assertion,
                       "' in function: '", funcname, "'\n"});
}

LIBC_INLINE void report_assertion_failure(cpp::string_view assertion,
                                          cpp::string_view filename,
                                          unsigned line,
                                          cpp::string_view funcname) {
  const IntegerToString<unsigned> line_buffer(line);
  report_assertion_failure(assertion, filename, line_buffer.view(), funcname);
}

// Direct calls to the unsafe_unreachable are highly discouraged.
// Use the macro versions to provide at least a check in debug builds.
LIBC_INLINE void unsafe_unreachable() {
#if __has_builtin(__builtin_unreachable)
  __builtin_unreachable();
#endif
}

// Direct calls to the unsafe_assume are highly discouraged.
// Use the macro versions to provide at least a check in debug builds.
LIBC_INLINE void unsafe_assume([[maybe_unused]] bool cond) {
#if __has_builtin(__builtin_assume)
  __builtin_assume(cond);
#elif __has_builtin(__builtin_unreachable)
  if (!cond)
    __builtin_unreachable();
#endif
}

} // namespace LIBC_NAMESPACE_DECL

// Macros:
// - LIBC_ASSERT(COND): similar to `assert(COND)` but may use libc's internal
// implementation.
// - LIBC_CHECK(COND): similar to LIBC_ASSERT(COND) but will not be disabled in
// release builds.
// - LIBC_ASSUME(COND): LIBC_CHECK in debug builds, __builtin_assume in release.
// - LIBC_UNREACHABLE(): similar to `__builtin_unreachable()` but checks in
// debug mode.
// - LIBC_CHECK_UNREACHABLE(): checks in both debug and release builds.

// Convert __LINE__ to a string using macros. The indirection is necessary
// because otherwise it will turn "__LINE__" into a string, not its value. The
// value is evaluated in the indirection step.
#define __LIBC_MACRO_TO_STR(x) #x
#define __LIBC_MACRO_TO_STR_INDIR(y) __LIBC_MACRO_TO_STR(y)
#define __LIBC_LINE_STR__ __LIBC_MACRO_TO_STR_INDIR(__LINE__)

#define LIBC_CHECK(COND)                                                       \
  do {                                                                         \
    if (!(COND)) {                                                             \
      LIBC_NAMESPACE::report_assertion_failure(                                \
          #COND, __FILE__, __LIBC_LINE_STR__, __PRETTY_FUNCTION__);            \
      LIBC_NAMESPACE::internal::exit(0xFF);                                    \
    }                                                                          \
  } while (false)

#if defined(LIBC_COPT_USE_C_ASSERT) || !defined(LIBC_FULL_BUILD)
// The build is configured to just use the public <assert.h> API
// for libc's internal assertions.

#include <assert.h>

#define LIBC_ASSERT(COND) assert(COND)

#else // Not LIBC_COPT_USE_C_ASSERT
#ifdef LIBC_ASSERT
#error "Unexpected: LIBC_ASSERT macro already defined"
#endif

// The public "assert" macro calls abort on failure. Should it be same here?
// The libc internal assert can fire from anywhere inside the libc. So, to
// avoid potential chicken-and-egg problems, it is simple to do an exit
// on assertion failure instead of calling abort. We also don't want to use
// __builtin_trap as it could potentially be implemented using illegal
// instructions which can be very misleading when debugging.
#ifdef NDEBUG
#define LIBC_ASSERT(COND)                                                      \
  do {                                                                         \
  } while (false)
#else
#define LIBC_ASSERT(COND) LIBC_CHECK(COND)
#endif // NDEBUG
#endif // LIBC_COPT_USE_C_ASSERT

#ifdef NDEBUG
#define LIBC_ASSUME() LIBC_NAMESPACE::unsafe_assume(true)
#else
#define LIBC_ASSUME(COND) LIBC_CHECK(COND)
#endif // NDEBUG

#define LIBC_CHECK_UNREACHABLE() LIBC_CHECK(false && "Unreachable code reached")

#ifdef NDEBUG
#define LIBC_UNREACHABLE() LIBC_NAMESPACE::unsafe_unreachable()
#else
#define LIBC_UNREACHABLE() LIBC_CHECK_UNREACHABLE()
#endif // NDEBUG

#endif // LLVM_LIBC_SRC___SUPPORT_LIBC_ASSERT_H
