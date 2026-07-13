//===-- Unittests for fmul ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MulTest.h"

#include "src/math/fmul.h"

LIST_MUL_TESTS(float, double, LIBC_NAMESPACE::fmul)

TEST_F(LlvmLibcMulTest, SpecialInputs) {
  constexpr double INPUTS[][2] = {
      {0x1.0100010002p8, 0x1.fffcp14},
      {0x1.000000b92144p-7, 0x1.62p7},
  };

  constexpr float RESULTS[] = {
      0x1.00fdfep+23f,
      0x1.620002p0f,
  };

  constexpr size_t N = sizeof(RESULTS) / sizeof(RESULTS[0]);

  for (size_t i = 0; i < N; ++i) {
    float result = LIBC_NAMESPACE::fmul(INPUTS[i][0], INPUTS[i][1]);
    EXPECT_FP_EQ(RESULTS[i], result);
  }
}
