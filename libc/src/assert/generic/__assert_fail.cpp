//===-- Implementation of __assert_fail -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/assert/__assert_fail.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/libc_assert.h"
#include "src/stdlib/abort.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, __assert_fail,
                   (const char *assertion, const char *file, unsigned line,
                    const char *function)) {
  __llvm_libc::report_assertion_failure(assertion, file, line, function);
  __llvm_libc::abort();
}

} // namespace __llvm_libc
