//===-- Definition of a libc internal assert macro --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_LIBC_ASSERT_H
#define LLVM_LIBC_SRC_SUPPORT_LIBC_ASSERT_H

#ifdef LIBC_COPT_USE_C_ASSERT

// The build is configured to just use the public <assert.h> API
// for libc's internal assertions.

#include <assert.h>

#define LIBC_ASSERT(COND) assert(COND)

#else // Not LIBC_COPT_USE_C_ASSERT

#include "src/__support/OSUtil/io.h"
#include "src/__support/OSUtil/quick_exit.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/attributes.h" // For LIBC_INLINE

namespace __llvm_libc {

LIBC_INLINE void report_assertion_failure(const char *assertion,
                                          const char *filename, unsigned line,
                                          const char *funcname) {
  char line_str[IntegerToString::dec_bufsize<unsigned>()];
  // dec returns an optional, will always be valid for this size buffer
  auto line_number = IntegerToString::dec(line, line_str);
  __llvm_libc::write_to_stderr(filename);
  __llvm_libc::write_to_stderr(":");
  __llvm_libc::write_to_stderr(*line_number);
  __llvm_libc::write_to_stderr(": Assertion failed: '");
  __llvm_libc::write_to_stderr(assertion);
  __llvm_libc::write_to_stderr("' in function: '");
  __llvm_libc::write_to_stderr(funcname);
  __llvm_libc::write_to_stderr("'\n");
}

} // namespace __llvm_libc

#ifdef LIBC_ASSERT
#error "Unexpected: LIBC_ASSERT macro already defined"
#endif

// The public "assert" macro calls abort on failure. Should it be same here?
// The libc internal assert can fire from anywhere inside the libc. So, to
// avoid potential chicken-and-egg problems, it is simple to do a quick_exit
// on assertion failure instead of calling abort. We also don't want to use
// __builtin_trap as it could potentially be implemented using illegal
// instructions which can be very misleading when debugging.
#ifdef NDEBUG
#define LIBC_ASSERT(COND)                                                      \
  do {                                                                         \
  } while (false)
#else
#define LIBC_ASSERT(COND)                                                      \
  do {                                                                         \
    if (!(COND)) {                                                             \
      __llvm_libc::report_assertion_failure(#COND, __FILE__, __LINE__,         \
                                            __PRETTY_FUNCTION__);              \
      __llvm_libc::quick_exit(0xFF);                                           \
    }                                                                          \
  } while (false)
#endif // NDEBUG

#endif // LIBC_COPT_USE_C_ASSERT

#endif // LLVM_LIBC_SRC_SUPPORT_LIBC_ASSERT_H
