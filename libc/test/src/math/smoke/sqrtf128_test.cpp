//===-- Unittests for sqrtf128---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SqrtTest.h"

#include "src/__support/FPUtil/float128.h"
#include "src/__support/integer_literals.h"
#include "src/math/sqrtf128.h"

#ifndef LIBC_TYPES_HAS_NATIVE_FLOAT128
using float128 = LIBC_NAMESPACE::fputil::Float128;
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128

using LIBC_NAMESPACE::operator""_u128;

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
      {FPBits(0x00000000'00dee2f5'b6a26c8f'07f05442_u128)
           .get_val(), // 0x0.000000dee2f5b6a26c8f07f05442p-16382q
       FPBits(0x9ff3ddbd'8763a617'cff753e2'a3108300_u128)
           .get_val()}, // -0x1.ddbd8763a617cff753e2a31083p-8204q
      {FPBits(0x00000000'00c86d17'4c5ad8ae'54a548e7_u128)
           .get_val(), // 0x0.000000c86d174c5ad8ae54a548e7p-16382q
       FPBits(0x1ff3c507'bb538940'71989085'1ec1ca88_u128)
           .get_val()}, // 0x1.c507bb538940719890851ec1ca88p-8204q
      {FPBits(0x00000000'20ab15cf'e0b8e488'e128f535_u128)
           .get_val(), // 0x0.000020ab15cfe0b8e488e128f535p-16382q
       FPBits(0x9ff66dcc'b4025602'13bc0d62'd62e910b_u128)
           .get_val()}, // -0x1.6dccb402560213bc0d62d62e910bp-8201q
      {FPBits(0x00000000'219e9773'2a9970f2'511989ba_u128)
           .get_val(), // 0x0.0000219e97732a9970f2511989bap-16382q
       FPBits(0x1ff67316'3d28be70'6f4b5052'791e28a5_u128)
           .get_val()}, // 0x1.73163d28be706f4b5052791e28a5p-8201q
      {FPBits(0x00000000'26e47754'6ae99ef5'7066f9fd_u128)
           .get_val(), // 0x0.000026e477546ae99ef57066f9fdp-16382q
       FPBits(0x9ff68f20'dd0d0c57'0a23ea59'bc2bf009_u128)
           .get_val()}, // -0x1.8f20dd0d0c570a23ea59bc2bf009p-8201q
      {FPBits(0x00000000'2d0f88d2'7a496b3e'533f5067_u128)
           .get_val(), // 0x0.00002d0f88d27a496b3e533f5067p-16382q
       FPBits(0x1ff6ad9d'4abe9f04'7225a735'2bcc52c1_u128)
           .get_val()}, // 0x1.ad9d4abe9f047225a7352bcc52c1p-8201q
      {FPBits(0x3fff0000'00000000'00000000'00000001_u128)
           .get_val(), // 0x1.0000000000000000000000000001p+0q
       FPBits(0x3fff0000'00000000'00000000'00000000_u128).get_val()}, // 0x1p+0q
      {FPBits(0x3fff0000'00000000'00000000'00000002_u128)
           .get_val(), // 0x1.0000000000000000000000000002p+0q
       FPBits(0xbfff0000'00000000'00000000'00000001_u128)
           .get_val()}, // -0x1.0000000000000000000000000001p+0q
      {FPBits(0x3fff0000'00000000'00000000'00000003_u128)
           .get_val(), // 0x1.0000000000000000000000000003p+0q
       FPBits(0x3fff0000'00000000'00000000'00000001_u128)
           .get_val()}, // 0x1.0000000000000000000000000001p+0q
      {FPBits(0x3fff0000'00000000'00000000'00000005_u128)
           .get_val(), // 0x1.0000000000000000000000000005p+0q
       FPBits(0x3fff0000'00000000'00000000'00000002_u128)
           .get_val()}, // 0x1.0000000000000000000000000002p+0q
      {FPBits(0x3fff0000'00000000'00000000'00000006_u128)
           .get_val(), // 0x1.0000000000000000000000000006p+0q
       FPBits(0xbfff0000'00000000'00000000'00000003_u128)
           .get_val()}, // -0x1.0000000000000000000000000003p+0q
      {FPBits(0x3fff1d4c'381cbf3a'0aa15b9a'ee344892_u128)
           .get_val(), // 0x1.1d4c381cbf3a0aa15b9aee344892p+0q
       FPBits(0x3fff0e40'8c3fadc5'e64b449c'63673f4b_u128)
           .get_val()}, // 0x1.0e408c3fadc5e64b449c63673f4bp+0q
      {FPBits(0x3fff2af1'7a4ae6f9'3d11310c'49c11b59_u128)
           .get_val(), // 0x1.2af17a4ae6f93d11310c49c11b59p+0q
       FPBits(0xbfff14a3'bdf0ea52'31f12d42'1a5dbe33_u128)
           .get_val()}, // -0x1.14a3bdf0ea5231f12d421a5dbe33p+0q
      {FPBits(0x3fff96f8'93bf29fb'91e0fbe1'9a46d0c8_u128)
           .get_val(), // 0x1.96f893bf29fb91e0fbe19a46d0c8p+0q
       FPBits(0x3fff42c6'bf6202e6'6f229580'7dee44d9_u128)
           .get_val()}, // 0x1.42c6bf6202e66f2295807dee44d9p+0q
      {FPBits(0x3fff97fb'3839925b'66804c42'9289cce8_u128)
           .get_val(), // 0x1.97fb3839925b66804c429289cce8p+0q
       FPBits(0xbfff432d'4049ac1c'85a241f3'33d326e9_u128)
           .get_val()}, // -0x1.432d4049ac1c85a241f333d326e9p+0q
      {FPBits(0x3fffbe1d'900eaeb1'533f0f19'cc15c7e6_u128)
           .get_val(), // 0x1.be1d900eaeb1533f0f19cc15c7e6p+0q
       FPBits(0x3fff51f1'715154da'44f3bf11'f3d96c2d_u128)
           .get_val()}, // 0x1.51f1715154da44f3bf11f3d96c2dp+0q
      {FPBits(0x3fffc4f5'07426952'5063a260'51a0ad27_u128)
           .get_val(), // 0x1.c4f5074269525063a26051a0ad27p+0q
       FPBits(0x3fff5486'4e9b1daa'4d9135ff'00663366_u128)
           .get_val()}, // 0x1.54864e9b1daa4d9135ff00663366p+0q
      {FPBits(0x4000035c'b5f298a8'01dc4be9'b1f8cd97_u128)
           .get_val(), // 0x1.035cb5f298a801dc4be9b1f8cd97p+1q
       FPBits(0xbfff6c68'8775bffc'b3f507ba'11d0abb9_u128)
           .get_val()}, // -0x1.6c688775bffcb3f507ba11d0abb9p+0q
      {FPBits(0x4000274b'e0238042'7e709bea'b4dedeb4_u128)
           .get_val(), // 0x1.274be02380427e709beab4dedeb4p+1q
       FPBits(0xbfff84d5'763281f2'31842239'2e506b1c_u128)
           .get_val()}, // -0x1.84d5763281f2318422392e506b1cp+0q
      {FPBits(0x400064e7'97cfdbaa'3f7e2f33'279dbc60_u128)
           .get_val(), // 0x1.64e797cfdbaa3f7e2f33279dbc6p+1q
       FPBits(0x3fffab79'b164e255'b26eca00'ff99cc99_u128)
           .get_val()}, // 0x1.ab79b164e255b26eca00ff99cc99p+0q
      {FPBits(0x4000693a'741358c9'dac44a57'0a7e9f6c_u128)
           .get_val(), // 0x1.693a741358c9dac44a570a7e9f6cp+1q
       FPBits(0x3fffae0e'8eaeab25'bb0c40ee'0c2693d3_u128)
           .get_val()}, // 0x1.ae0e8eaeab25bb0c40ee0c2693d3p+0q
      {FPBits(0x40008275'db3fc4d8'22596047'adcb71b9_u128)
           .get_val(), // 0x1.8275db3fc4d822596047adcb71b9p+1q
       FPBits(0xbfffbcd2'bfb653e3'7a5dbe0c'cc2cd917_u128)
           .get_val()}, // -0x1.bcd2bfb653e37a5dbe0ccc2cd917p+0q
      {FPBits(0x40008328'0bb98c4a'7b88bd6f'535899d9_u128)
           .get_val(), // 0x1.83280bb98c4a7b88bd6f535899d9p+1q
       FPBits(0x3fffbd39'409dfd19'90dd6a7f'8211bb27_u128)
           .get_val()}, // 0x1.bd39409dfd1990dd6a7f8211bb27p+0q
      {FPBits(0x4000d78d'8352b486'08b510bf'd5c75315_u128)
           .get_val(), // 0x1.d78d8352b48608b510bfd5c75315p+1q
       FPBits(0xbfffeb5c'420f15ad'ce0ed2bd'e5a241ce_u128)
           .get_val()}, // -0x1.eb5c420f15adce0ed2bde5a241cep+0q
      {FPBits(0x4000e3e4'774f564b'526edff8'4ce46668_u128)
           .get_val(), // 0x1.e3e4774f564b526edff84ce46668p+1q
       FPBits(0x3ffff1bf'73c0523a'19b4bb63'9c98c0b5_u128)
           .get_val()}, // 0x1.f1bf73c0523a19b4bb639c98c0b5p+0q
      {FPBits(0x4000ffff'ffffffff'ffffffff'fffffffa_u128)
           .get_val(), // 0x1.fffffffffffffffffffffffffffap+1q
       FPBits(0xbfffffff'ffffffff'ffffffff'fffffffd_u128)
           .get_val()}, // -0x1.fffffffffffffffffffffffffffdp+0q
      {FPBits(0x4000ffff'ffffffff'ffffffff'fffffffb_u128)
           .get_val(), // 0x1.fffffffffffffffffffffffffffbp+1q
       FPBits(0x3fffffff'ffffffff'ffffffff'fffffffd_u128)
           .get_val()}, // 0x1.fffffffffffffffffffffffffffdp+0q
      {FPBits(0x4000ffff'ffffffff'ffffffff'fffffffd_u128)
           .get_val(), // 0x1.fffffffffffffffffffffffffffdp+1q
       FPBits(0x3fffffff'ffffffff'ffffffff'fffffffe_u128)
           .get_val()}, // 0x1.fffffffffffffffffffffffffffep+0q
      {FPBits(0x4000ffff'ffffffff'ffffffff'fffffffe_u128)
           .get_val(), // 0x1.fffffffffffffffffffffffffffep+1q
       FPBits(0xbfffffff'ffffffff'ffffffff'ffffffff_u128)
           .get_val()}, // -0x1.ffffffffffffffffffffffffffffp+0q
      {FPBits(0x4000ffff'ffffffff'ffffffff'ffffffff_u128)
           .get_val(), // 0x1.ffffffffffffffffffffffffffffp+1q
       FPBits(0x3fffffff'ffffffff'ffffffff'ffffffff_u128)
           .get_val()}, // 0x1.ffffffffffffffffffffffffffffp+0q
  };

  auto rnd = [](float128 x, RoundingMode rm) -> float128 {
    bool is_neg = x < float128(0);
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
      {FPBits(0x00000000'00000000'00000000'00000001_u128)
           .get_val(), // 0x0.0000000000000000000000000001p-16382q
       FPBits(0x1fc80000'00000000'00000000'00000000_u128)
           .get_val()}, // 0x1p-8247q
      {FPBits(0x00000000'00000000'00000000'00000004_u128)
           .get_val(), // 0x0.0000000000000000000000000004p-16382q
       FPBits(0x1fc90000'00000000'00000000'00000000_u128)
           .get_val()}, // 0x1p-8246q
      {FPBits(0x00000000'00000000'10000000'00000000_u128)
           .get_val(), // 0x0.0000000000001000000000000000p-16382q
       FPBits(0x1fe60000'00000000'00000000'00000000_u128)
           .get_val()}, // 0x1p-8217q
      {FPBits(0x00000000'00000001'00000000'00000000_u128)
           .get_val(), // 0x0.0000000000010000000000000000p-16382q
       FPBits(0x1fe80000'00000000'00000000'00000000_u128)
           .get_val()}, // 0x1p-8215q
      {FPBits(0x00000000'00000010'00000000'00000000_u128)
           .get_val(), // 0x0.0000000000100000000000000000p-16382q
       FPBits(0x1fea0000'00000000'00000000'00000000_u128)
           .get_val()}, // 0x1p-8213q
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
    float128 k_f128 = static_cast<float128>(k);
    float128 x = k_f128 * k_f128;
    float128 y = static_cast<float128>(k);
    EXPECT_FP_EQ_ALL_ROUNDING(y, LIBC_NAMESPACE::sqrtf128(x));
  }
}
