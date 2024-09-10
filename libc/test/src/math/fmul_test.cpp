//===-- Unittests for fmul ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MulTest.h"

#include "src/math/fmul.h"

#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

LIST_MUL_TESTS(float, double, LIBC_NAMESPACE::fmul)

TEST_F(LlvmLibcMulTest, SpecialInputs) {
  namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
  double INPUTS[][2] = {
      {0x1.0100010002p8, 0x1.fffcp14},
  };

  constexpr float RESULTS[] = {
      0x1.00fdfep+23f,
  };

  
  constexpr size_t N = sizeof(RESULTS) / sizeof(RESULTS[0]);

  for (size_t i = 0; i < N; ++i) {
    double a = INPUTS[i][0];

    for (int j = 0; j < 180; ++j) {
      a *= 0.5;
      mpfr::BinaryInput<double> input{INPUTS[i][0], INPUTS[i][1]};
      ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Mul, input, LIBC_NAMESPACE::fmul(INPUTS[i][0], INPUTS[i][1]), 0.5);
    }
  }
}
