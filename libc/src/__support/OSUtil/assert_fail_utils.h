//===-- Assertion failure reporting helper ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_ASSERT_FAIL_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_ASSERT_FAIL_UTILS_H

#include "src/__support/OSUtil/io.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

LIBC_INLINE void report_assertion_failure(const char *assertion,
                                          const char *filename, unsigned line,
                                          const char *funcname) {
  const IntegerToString<unsigned> line_buffer(line);
  write_to_stderr(filename);
  write_to_stderr(":");
  write_to_stderr(line_buffer.view());
  write_to_stderr(": Assertion failed: '");
  write_to_stderr(assertion);
  write_to_stderr("' in function: '");
  write_to_stderr(funcname);
  write_to_stderr("'\n");
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_ASSERT_FAIL_UTILS_H
