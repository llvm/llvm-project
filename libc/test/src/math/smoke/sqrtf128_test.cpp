//===-- Unittests for sqrtf128---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SqrtTest.h"

#include "src/__support/uint128.h"
#include "src/math/sqrtf128.h"

LIST_SQRT_TESTS(float128, LIBC_NAMESPACE::sqrtf128);

TEST_F(LlvmLibcSqrtTest, HardToRound) {
  using LIBC_NAMESPACE::fputil::testing::RoundingMode;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float128>;

  // Since there is no exact half cases for square root I encode the
  // round direction in the sign of the result. E.g. if the number is
  // negative it means that the exact root is below the rounded value
  // (the absolute value). Thus I can test not only hard to round
  // cases for the round to nearest mode but also the directional
  // modes.
  float128 HARD_TO_ROUND[][2] = {
      {0x0.000000dee2f5b6a26c8f07f05442p-16382q,
       -0x1.ddbd8763a617cff753e2a31083p-8204q},
      {0x0.000000c86d174c5ad8ae54a548e7p-16382q,
       0x1.c507bb538940719890851ec1ca88p-8204q},
      {0x0.000020ab15cfe0b8e488e128f535p-16382q,
       -0x1.6dccb402560213bc0d62d62e910bp-8201q},
      {0x0.0000219e97732a9970f2511989bap-16382q,
       0x1.73163d28be706f4b5052791e28a5p-8201q},
      {0x0.000026e477546ae99ef57066f9fdp-16382q,
       -0x1.8f20dd0d0c570a23ea59bc2bf009p-8201q},
      {0x0.00002d0f88d27a496b3e533f5067p-16382q,
       0x1.ad9d4abe9f047225a7352bcc52c1p-8201q},
      {0x1.0000000000000000000000000001p+0q, 0x1p+0q},
      {0x1.0000000000000000000000000002p+0q,
       -0x1.0000000000000000000000000001p+0q},
      {0x1.0000000000000000000000000003p+0q,
       0x1.0000000000000000000000000001p+0q},
      {0x1.0000000000000000000000000005p+0q,
       0x1.0000000000000000000000000002p+0q},
      {0x1.0000000000000000000000000006p+0q,
       -0x1.0000000000000000000000000003p+0q},
      {0x1.1d4c381cbf3a0aa15b9aee344892p+0q,
       0x1.0e408c3fadc5e64b449c63673f4bp+0q},
      {0x1.2af17a4ae6f93d11310c49c11b59p+0q,
       -0x1.14a3bdf0ea5231f12d421a5dbe33p+0q},
      {0x1.96f893bf29fb91e0fbe19a46d0c8p+0q,
       0x1.42c6bf6202e66f2295807dee44d9p+0q},
      {0x1.97fb3839925b66804c429289cce8p+0q,
       -0x1.432d4049ac1c85a241f333d326e9p+0q},
      {0x1.be1d900eaeb1533f0f19cc15c7e6p+0q,
       0x1.51f1715154da44f3bf11f3d96c2dp+0q},
      {0x1.c4f5074269525063a26051a0ad27p+0q,
       0x1.54864e9b1daa4d9135ff00663366p+0q},
      {0x1.035cb5f298a801dc4be9b1f8cd97p+1q,
       -0x1.6c688775bffcb3f507ba11d0abb9p+0q},
      {0x1.274be02380427e709beab4dedeb4p+1q,
       -0x1.84d5763281f2318422392e506b1cp+0q},
      {0x1.64e797cfdbaa3f7e2f33279dbc6p+1q,
       0x1.ab79b164e255b26eca00ff99cc99p+0q},
      {0x1.693a741358c9dac44a570a7e9f6cp+1q,
       0x1.ae0e8eaeab25bb0c40ee0c2693d3p+0q},
      {0x1.8275db3fc4d822596047adcb71b9p+1q,
       -0x1.bcd2bfb653e37a5dbe0ccc2cd917p+0q},
      {0x1.83280bb98c4a7b88bd6f535899d9p+1q,
       0x1.bd39409dfd1990dd6a7f8211bb27p+0q},
      {0x1.d78d8352b48608b510bfd5c75315p+1q,
       -0x1.eb5c420f15adce0ed2bde5a241cep+0q},
      {0x1.e3e4774f564b526edff84ce46668p+1q,
       0x1.f1bf73c0523a19b4bb639c98c0b5p+0q},
      {0x1.fffffffffffffffffffffffffffap+1q,
       -0x1.fffffffffffffffffffffffffffdp+0q},
      {0x1.fffffffffffffffffffffffffffbp+1q,
       0x1.fffffffffffffffffffffffffffdp+0q},
      {0x1.fffffffffffffffffffffffffffdp+1q,
       0x1.fffffffffffffffffffffffffffep+0q},
      {0x1.fffffffffffffffffffffffffffep+1q,
       -0x1.ffffffffffffffffffffffffffffp+0q},
      {0x1.ffffffffffffffffffffffffffffp+1q,
       0x1.ffffffffffffffffffffffffffffp+0q},
  };

  auto rnd = [](float128 x, RoundingMode rm) -> float128 {
    bool is_neg = x < 0;
    float128 y = is_neg ? -x : x;
    FPBits ybits(y);

    if (is_neg &&
        (rm == RoundingMode::Downward || rm == RoundingMode::TowardZero))
      return FPBits(ybits.uintval() - 1).get_val();
    if (!is_neg && (rm == RoundingMode::Upward))
      return FPBits(ybits.uintval() + 1).get_val();

    return y;
  };

  for (auto &t : HARD_TO_ROUND) {
    EXPECT_FP_EQ_ALL_ROUNDING(
        rnd(t[1], RoundingMode::Nearest), rnd(t[1], RoundingMode::Upward),
        rnd(t[1], RoundingMode::Downward), rnd(t[1], RoundingMode::TowardZero),
        LIBC_NAMESPACE::sqrtf128(t[0]));
  }

  // Exact results for subnormal arguments
  float128 EXACT_SUBNORMAL[][2] = {
      {0x0.0000000000000000000000000001p-16382q, 0x1p-8247q},
      {0x0.0000000000000000000000000004p-16382q, 0x1p-8246q},
      {0x0.0000000000001000000000000000p-16382q, 0x1p-8217q},
      {0x0.0000000000010000000000000000p-16382q, 0x1p-8215q},
      {0x0.0000000000100000000000000000p-16382q, 0x1p-8213q},
  };

  for (auto t : EXACT_SUBNORMAL)
    EXPECT_FP_EQ_ALL_ROUNDING(t[1], LIBC_NAMESPACE::sqrtf128(t[0]));

  // Check exact cases starting from small numbers
  for (unsigned k = 1; k < 100 * 100; ++k) {
    unsigned k2 = k * k;
    float128 x = static_cast<float128>(k2);
    float128 y = static_cast<float128>(k);
    EXPECT_FP_EQ_ALL_ROUNDING(y, LIBC_NAMESPACE::sqrtf128(x));
  };

  // Then from the largest number.
  uint64_t k0 = 101904826760412362ULL;
  for (uint64_t k = k0; k > k0 - 10000; --k) {
    UInt128 k2 = static_cast<UInt128>(k) * static_cast<UInt128>(k);
    float128 x = static_cast<float128>(k2);
    float128 y = static_cast<float128>(k);
    EXPECT_FP_EQ_ALL_ROUNDING(y, LIBC_NAMESPACE::sqrtf128(x));
  }
}
