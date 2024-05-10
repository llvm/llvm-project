//===-- Header for using the gtest framework -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_GTEST_H
#define LLVM_LIBC_UTILS_UNITTEST_GTEST_H

#include <gtest/gtest.h>

namespace LIBC_NAMESPACE::testing {

using ::testing::Matcher;
using ::testing::Test;

} // namespace LIBC_NAMESPACE::testing

#define LIBC_TEST_HAS_MATCHERS() (1)

#endif // LLVM_LIBC_UTILS_UNITTEST_GTEST_H
