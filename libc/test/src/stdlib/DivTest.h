//===-- A template class for testing div functions --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

class LlvmLibcDivTest : public LIBC_NAMESPACE::testing::Test {
public:
  template <typename IntType, typename ReturnType>
  void simpleTest(ReturnType (&func)(IntType, IntType)) {
    auto result = func(10, 3);
    EXPECT_EQ(result.quot, IntType(3));
    EXPECT_EQ(result.rem, IntType(1));

    result = func(-10, 3);
    EXPECT_EQ(result.quot, IntType(-3));
    EXPECT_EQ(result.rem, IntType(-1));

    result = func(-10, -3);
    EXPECT_EQ(result.quot, IntType(3));
    EXPECT_EQ(result.rem, IntType(-1));

    result = func(10, -3);
    EXPECT_EQ(result.quot, IntType(-3));
    EXPECT_EQ(result.rem, IntType(1));
  }
};

#define LIST_DIV_TESTS(IntType, ReturnType, func)                              \
  TEST_F(LlvmLibcDivTest, SimpleTest##ReturnType) {                            \
    simpleTest<IntType, ReturnType>(func);                                     \
  }
