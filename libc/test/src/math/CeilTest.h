//===-- Utility class to test ceil[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_CEILTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_CEILTEST_H

#include "src/__support/CPP/algorithm.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include "hdr/math_macros.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename T>
class CeilTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*CeilFunc)(T);

  void testRange(CeilFunc func) {
    constexpr int COUNT = 1'231;
    constexpr StorageType STEP = LIBC_NAMESPACE::cpp::max(
        static_cast<StorageType>(STORAGE_MAX / COUNT), StorageType(1));
    StorageType v = 0;
    for (int i = 0; i <= COUNT; ++i, v += STEP) {
      FPBits xbits(v);
      T x = xbits.get_val();
      if (xbits.is_inf_or_nan())
        continue;

      ASSERT_MPFR_MATCH(mpfr::Operation::Ceil, x, func(x), 0.0);
    }
  }
};

#define LIST_CEIL_TESTS(T, func)                                               \
  using LlvmLibcCeilTest = CeilTest<T>;                                        \
  TEST_F(LlvmLibcCeilTest, Range) { testRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_CEILTEST_H
