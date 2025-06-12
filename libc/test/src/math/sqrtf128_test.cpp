//===-- Unittests for sqrtf128 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SqrtTest.h"

#include "src/math/sqrtf128.h"

#include "src/__support/integer_literals.h"

LIST_SQRT_TESTS(float128, LIBC_NAMESPACE::sqrtf128)

TEST_F(LlvmLibcSqrtTest, SpecialInputs) {
  constexpr float128 INPUTS[] = {
      0x0.000000dee2f5b6a26c8f07f05442p-16382q,
      0x0.000000c86d174c5ad8ae54a548e7p-16382q,
      0x0.000020ab15cfe0b8e488e128f535p-16382q,
      0x0.0000219e97732a9970f2511989bap-16382q,
      0x0.000026e477546ae99ef57066f9fdp-16382q,
      0x0.00002d0f88d27a496b3e533f5067p-16382q,
      0x1.0000000000000000000000000001p+0q,
      0x1.0000000000000000000000000003p+0q,
      0x1.0000000000000000000000000005p+0q,
      0x1.2af17a4ae6f93d11310c49c11b59p+0q,
      0x1.c4f5074269525063a26051a0ad27p+0q,
      0x1.035cb5f298a801dc4be9b1f8cd97p+1q,
      0x1.274be02380427e709beab4dedeb4p+1q,
      0x1.64e797cfdbaa3f7e2f33279dbc6p+1q,
      0x1.d78d8352b48608b510bfd5c75315p+1q,
      0x1.fffffffffffffffffffffffffffbp+1q,
      0x1.fffffffffffffffffffffffffffdp+1q,
      0x1.ffffffffffffffffffffffffffffp+1q,
  };

  for (auto input : INPUTS) {
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sqrt, input,
                                   LIBC_NAMESPACE::sqrtf128(input), 0.5);
  }
}
