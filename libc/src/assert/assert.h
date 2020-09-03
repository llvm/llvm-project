//===-- Internal header for assert ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ASSERT_ASSERT_H
#define LLVM_LIBC_SRC_ASSERT_ASSERT_H

#include <stddef.h>

namespace __llvm_libc {

[[noreturn]] void __assert_fail(const char *assertion, const char *file, unsigned line,
                  const char *function);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_ASSERT_ASSERT_H

#undef assert

#ifdef NDEBUG
#define assert(e) (void)0
#else
#define assert(e)                                                              \
  ((e) ? (void)0 :                                                             \
    __llvm_libc::__assert_fail(#e, __FILE__, __LINE__, __PRETTY_FUNCTION__))
#endif
