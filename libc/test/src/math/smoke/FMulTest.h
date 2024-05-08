//===-- Utility class to test fmul[f|l] ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMULTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMULTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T, typename R>
class FmulTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

public:
  typedef T (*FMulFunc)(R, R);

  void testMul(FMulFunc func) { EXPECT_FP_EQ(T(1.0), func(1.0, 1.0)); }
};

#define LIST_FMUL_TESTS(T, R, func)                                            \
  using LlvmLibcFmulTest = FmulTest<T, R>;                                     \
  TEST_F(LlvmLibcFmulTest, Mul) { testMul(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMULTEST_H
