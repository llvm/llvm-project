//===-- Header for setting up the Fuchsia tests -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_FUCHSIATEST_H
#define LLVM_LIBC_UTILS_UNITTEST_FUCHSIATEST_H

#include <zxtest/zxtest.h>

#define WITH_SIGNAL(X) #X

// These macros are used in string unittests.
#define ASSERT_ERRNO_EQ(VAL) ASSERT_EQ(VAL, static_cast<int>(libc_errno))
#define ASSERT_ERRNO_SUCCESS() ASSERT_EQ(0, static_cast<int>(libc_errno))
#define ASSERT_ERRNO_FAILURE() ASSERT_NE(0, static_cast<int>(libc_errno))

#ifndef EXPECT_DEATH
// Since zxtest has ASSERT_DEATH but not EXPECT_DEATH, wrap calling it
// in a lambda returning void to swallow any early returns so that this
// can be used in a function that itself returns non-void.
#define EXPECT_DEATH(FUNC, SIG) ([&] { ASSERT_DEATH(FUNC, SIG); }())
#endif

namespace LIBC_NAMESPACE::testing {
using Test = ::zxtest::Test;
}

#endif // LLVM_LIBC_UTILS_UNITTEST_FUCHSIATEST_H
