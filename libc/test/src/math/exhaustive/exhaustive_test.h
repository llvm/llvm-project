//===-- Exhaustive test template for math functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/TypeTraits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

// To test exhaustively for inputs in the range [start, stop) in parallel:
// 1. Inherit from LlvmLibcExhaustiveTest class
// 2. Overide the test method: void check(T, T, RoundingMode)
// 4. Call: test_full_range(start, stop, nthreads, rounding)
namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T, typename FloatType = float>
struct LlvmLibcExhaustiveTest : public __llvm_libc::testing::Test {
  static constexpr T increment = (1 << 20);
  static_assert(
      __llvm_libc::cpp::IsSameV<
          T, typename __llvm_libc::fputil::FPBits<FloatType>::UIntType>,
      "Types are not consistent");
  // Break [start, stop) into `nthreads` subintervals and apply *check to each
  // subinterval in parallel.
  void test_full_range(T start, T stop, mpfr::RoundingMode rounding);

  virtual bool check(T start, T stop, mpfr::RoundingMode rounding) = 0;
};
