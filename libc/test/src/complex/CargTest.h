//===-- Utility class to test different flavors of carg ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_COMPLEX_CARGTEST_H
#define LLVM_LIBC_TEST_SRC_COMPLEX_CARGTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/math_macros.h"

template <typename CFPT, typename FPT>
class CargTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(FPT)

public:
  typedef FPT (*CargFunc)(CFPT);

  void testRoundedNumbers(CargFunc func) {
    EXPECT_FP_EQ((FPT)(1.10714871762320399284), func((CFPT)(1.0 + 2.0i)));
  }
};

#define LIST_CARG_TESTS(U, T, func)                                          \
  using LlvmLibcCargTest = CargTest<U, T>;                                   \
  TEST_F(LlvmLibcCargTest, RoundedNumbers) { testRoundedNumbers(&func); }

#endif // LLVM_LIBC_TEST_SRC_COMPLEX_CARGTEST_H
