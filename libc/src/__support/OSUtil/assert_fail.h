//===-- Internal header for assert_fail -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_ASSERT_FAIL_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_ASSERT_FAIL_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

[[noreturn]] void assert_fail(const char *assertion, const char *file,
                              unsigned line, const char *function);

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_ASSERT_FAIL_H
