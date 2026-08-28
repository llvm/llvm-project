//===-- Unittests for sqrtf128 --------------------------------------------===//
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

using LIBC_NAMESPACE::operator""_u128;

#ifndef LIBC_TYPES_HAS_NATIVE_FLOAT128
using float128 = LIBC_NAMESPACE::fputil::Float128;
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128

LIST_SQRT_TESTS(float128, LIBC_NAMESPACE::sqrtf128)

TEST_F(LlvmLibcSqrtTest, SpecialInputs) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float128>;

  float128 INPUTS[] = {
      FPBits(0x00000000'00dee2f5'b6a26c8f'07f05442_u128)
          .get_val(), // 0x0.000000dee2f5b6a26c8f07f05442p-16382q
      FPBits(0x00000000'00c86d17'4c5ad8ae'54a548e7_u128)
          .get_val(), // 0x0.000000c86d174c5ad8ae54a548e7p-16382q
      FPBits(0x00000000'20ab15cf'e0b8e488'e128f535_u128)
          .get_val(), // 0x0.000020ab15cfe0b8e488e128f535p-16382q
      FPBits(0x00000000'219e9773'2a9970f2'511989ba_u128)
          .get_val(), // 0x0.0000219e97732a9970f2511989bap-16382q
      FPBits(0x00000000'26e47754'6ae99ef5'7066f9fd_u128)
          .get_val(), // 0x0.000026e477546ae99ef57066f9fdp-16382q
      FPBits(0x00000000'2d0f88d2'7a496b3e'533f5067_u128)
          .get_val(), // 0x0.00002d0f88d27a496b3e533f5067p-16382q
      FPBits(0x3fff0000'00000000'00000000'00000001_u128)
          .get_val(), // 0x1.0000000000000000000000000001p+0q
      FPBits(0x3fff0000'00000000'00000000'00000003_u128)
          .get_val(), // 0x1.0000000000000000000000000003p+0q
      FPBits(0x3fff0000'00000000'00000000'00000005_u128)
          .get_val(), // 0x1.0000000000000000000000000005p+0q
      FPBits(0x3fff2af1'7a4ae6f9'3d11310c'49c11b59_u128)
          .get_val(), // 0x1.2af17a4ae6f93d11310c49c11b59p+0q
      FPBits(0x3fffc4f5'07426952'5063a260'51a0ad27_u128)
          .get_val(), // 0x1.c4f5074269525063a26051a0ad27p+0q
      FPBits(0x4000035c'b5f298a8'01dc4be9'b1f8cd97_u128)
          .get_val(), // 0x1.035cb5f298a801dc4be9b1f8cd97p+1q
      FPBits(0x4000274b'e0238042'7e709bea'b4dedeb4_u128)
          .get_val(), // 0x1.274be02380427e709beab4dedeb4p+1q
      FPBits(0x400064e7'97cfdbaa'3f7e2f33'279dbc60_u128)
          .get_val(), // 0x1.64e797cfdbaa3f7e2f33279dbc6p+1q
      FPBits(0x4000d78d'8352b486'08b510bf'd5c75315_u128)
          .get_val(), // 0x1.d78d8352b48608b510bfd5c75315p+1q
      FPBits(0x4000ffff'ffffffff'ffffffff'fffffffb_u128)
          .get_val(), // 0x1.fffffffffffffffffffffffffffbp+1q
      FPBits(0x4000ffff'ffffffff'ffffffff'fffffffd_u128)
          .get_val(), // 0x1.fffffffffffffffffffffffffffdp+1q
      FPBits(0x4000ffff'ffffffff'ffffffff'ffffffff_u128)
          .get_val(), // 0x1.ffffffffffffffffffffffffffffp+1q
  };

  for (auto input : INPUTS) {
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sqrt, input,
                                   LIBC_NAMESPACE::sqrtf128(input), 0.5);
  }
}
