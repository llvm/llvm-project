//===-- ErrnoCheckingTest.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_ERRNOCHECKINGTEST_H
#define LLVM_LIBC_TEST_UNITTEST_ERRNOCHECKINGTEST_H

#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"
#include "test/UnitTest/Test.h"

// Define macro to validate the value stored in the errno and restore it
// to zero.

#ifdef LIBC_TARGET_ARCH_IS_GPU
#define ASSERT_ERRNO_EQ(VAL)
#define ASSERT_ERRNO_SUCCESS()
#define ASSERT_ERRNO_FAILURE()
#else
#define ASSERT_ERRNO_EQ(VAL)                                                   \
  do {                                                                         \
    ASSERT_EQ(VAL, static_cast<int>(libc_errno));                              \
    libc_errno = 0;                                                            \
  } while (0)
#define ASSERT_ERRNO_SUCCESS() ASSERT_EQ(0, static_cast<int>(libc_errno))
#define ASSERT_ERRNO_FAILURE()                                                 \
  do {                                                                         \
    ASSERT_NE(0, static_cast<int>(libc_errno));                                \
    libc_errno = 0;                                                            \
  } while (0)
#endif

namespace LIBC_NAMESPACE_DECL {
namespace testing {

// Provides a test fixture for tests that validate modifications of the errno.
// It clears out the errno at the beginning of the test (e.g. in case it
// contained the value pre-set by the system), and confirms it's still zero
// at the end of the test, forcing the test author to explicitly account for all
// non-zero values.
class ErrnoCheckingTest : public Test {
public:
  void SetUp() override {
    Test::SetUp();
    libc_errno = 0;
  }

  void TearDown() override {
    ASSERT_ERRNO_SUCCESS();
    Test::TearDown();
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_TEST_UNITTEST_ERRNOCHECKINGTEST_H
