//===-- Windows implementation of internal::assert_fail ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/assert_fail.h"
#include "src/__support/OSUtil/assert_fail_utils.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

[[noreturn]] void assert_fail(const char *assertion, const char *file,
                              unsigned line, const char *function) {
  report_assertion_failure(assertion, file, line, function);
  internal::exit(0xFF);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
