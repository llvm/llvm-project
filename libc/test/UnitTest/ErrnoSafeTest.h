//===-- ErrnoSafeTestFixture.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_ERRNOSAFETEST_H
#define LLVM_LIBC_TEST_UNITTEST_ERRNOSAFETEST_H

#include "src/__support/CPP/utility.h"
#include "src/errno/libc_errno.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE::testing {

// This is a test fixture for clearing errno before the start of a test case.
class ErrnoSafeTest : virtual public Test {
public:
  void SetUp() override { LIBC_NAMESPACE::libc_errno = 0; }
};

} // namespace LIBC_NAMESPACE::testing

#endif // LLVM_LIBC_TEST_UNITTEST_ERRNOSAFETEST_H
