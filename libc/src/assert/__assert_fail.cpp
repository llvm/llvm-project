//===-- Implementation of __assert_fail -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/assert/__assert_fail.h"
#include "src/stdlib/abort.h"

// These includes are temporary.
#include "config/linux/syscall.h" // For internal syscall function.
#include "include/sys/syscall.h"  // For syscall numbers.

namespace __llvm_libc {

// This is just a temporary solution to make assert available to internal
// llvm libc code. In the future writeToStderr will not exist and __assert_fail
// will call fprintf(stderr, ...).
static void writeToStderr(const char *s) {
  size_t length = 0;
  for (const char *curr = s; *curr; ++curr, ++length);
  __llvm_libc::syscall(SYS_write, 2, s, length);
}

void LLVM_LIBC_ENTRYPOINT(__assert_fail)(const char *assertion, const char *file,
                                         unsigned line, const char *function) {
  writeToStderr(file);
  writeToStderr(": Assertion failed: '");
  writeToStderr(assertion);
  writeToStderr("' in function: '");
  writeToStderr(function);
  writeToStderr("'\n");
  __llvm_libc::abort();
}

} // namespace __llvm_libc
