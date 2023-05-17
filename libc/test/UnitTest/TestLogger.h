//===-- Utilities to log to standard output during tests --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_TESTLOGGER_H
#define LLVM_LIBC_TEST_UNITTEST_TESTLOGGER_H

namespace __llvm_libc {
namespace testing {

// A class to log to standard output in the context of hermetic tests.
struct TestLogger {
  constexpr TestLogger() = default;
  template <typename T> TestLogger &operator<<(T);
};

// A global TestLogger instance to be used in tests.
extern TestLogger tlog;

} // namespace testing
} // namespace __llvm_libc

#endif /* LLVM_LIBC_TEST_UNITTEST_TESTLOGGER_H */
