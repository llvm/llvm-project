//===-- Implementation of __assert_fail -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/assert/__assert_fail.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/abort_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, __assert_fail,
                   (const char *assertion, const char *file, unsigned line,
                    const char *function)) {
  // Re-entrancy guard: if an assertion fails inside the abort or lock sequence,
  // trap immediately to prevent infinite recursion.
  static cpp::Atomic<bool> in_assertion{false};
  if (in_assertion.exchange(true, cpp::MemoryOrder::ACQUIRE))
    __builtin_trap();

  LIBC_NAMESPACE::report_assertion_failure(assertion, file, line, function);
  LIBC_NAMESPACE::abort_utils::abort();
}

} // namespace LIBC_NAMESPACE_DECL
