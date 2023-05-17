//===-- Unittests for high_precision_decimal ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/UInt128.h"
#include "src/__support/high_precision_decimal.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcHighPrecisionDecimalTest, BasicInit) {
  __llvm_libc::internal::HighPrecisionDecimal hpd =
      __llvm_libc::internal::HighPrecisionDecimal("1.2345");
  uint8_t *digits = hpd.get_digits();

  EXPECT_EQ(digits[0], uint8_t(1));
  EXPECT_EQ(digits[1], uint8_t(2));
  EXPECT_EQ(digits[2], uint8_t(3));
  EXPECT_EQ(digits[3], uint8_t(4));
  EXPECT_EQ(digits[4], uint8_t(5));
  EXPECT_EQ(hpd.get_num_digits(), 5u);
  EXPECT_EQ(hpd.get_decimal_point(), 1);
}

TEST(LlvmLibcHighPrecisionDecimalTest, BasicShift) {
  __llvm_libc::internal::HighPrecisionDecimal hpd =
      __llvm_libc::internal::HighPrecisionDecimal("1");
  uint8_t *digits = hpd.get_digits();

  hpd.shift(1); // shift left 1, equal to multiplying by 2.

  EXPECT_EQ(digits[0], uint8_t(2));
  EXPECT_EQ(hpd.get_num_digits(), 1u);
  EXPECT_EQ(hpd.get_decimal_point(), 1);
}

TEST(LlvmLibcHighPrecisionDecimalTest, ShouldRoundup) {
  __llvm_libc::internal::HighPrecisionDecimal hpd =
      __llvm_libc::internal::HighPrecisionDecimal(".5");
  uint8_t *digits = hpd.get_digits();

  EXPECT_EQ(digits[0], uint8_t(5));
  EXPECT_EQ(hpd.get_num_digits(), 1u);
  EXPECT_EQ(hpd.get_decimal_point(), 0);
  EXPECT_EQ(hpd.round_to_integer_type<int>(), 0);
}

TEST(LlvmLibcHighPrecisionDecimalTest, SmallShift) {
  __llvm_libc::internal::HighPrecisionDecimal hpd =
      __llvm_libc::internal::HighPrecisionDecimal("1.2345");
  uint8_t *digits = hpd.get_digits();

  hpd.shift(-1); // shift right one, equal to dividing by 2
  // result should be 0.61725

  EXPECT_EQ(digits[0], uint8_t(6));
  EXPECT_EQ(digits[1], uint8_t(1));
  EXPECT_EQ(digits[2], uint8_t(7));
  EXPECT_EQ(digits[3], uint8_t(2));
  EXPECT_EQ(digits[4], uint8_t(5));
  EXPECT_EQ(hpd.get_num_digits(), 5u);
  EXPECT_EQ(hpd.get_decimal_point(), 0);

  hpd.shift(1); // shift left one, equal to multiplying by 2
  // result should be 1.2345 again

  EXPECT_EQ(digits[0], uint8_t(1));
  EXPECT_EQ(digits[1], uint8_t(2));
  EXPECT_EQ(digits[2], uint8_t(3));
  EXPECT_EQ(digits[3], uint8_t(4));
  EXPECT_EQ(digits[4], uint8_t(5));
  EXPECT_EQ(hpd.get_num_digits(), 5u);
  EXPECT_EQ(hpd.get_decimal_point(), 1);

  hpd.shift(1); // shift left one again
  // result should be 2.469

  EXPECT_EQ(digits[0], uint8_t(2));
  EXPECT_EQ(digits[1], uint8_t(4));
  EXPECT_EQ(digits[2], uint8_t(6));
  EXPECT_EQ(digits[3], uint8_t(9));
  EXPECT_EQ(hpd.get_num_digits(), 4u);
  EXPECT_EQ(hpd.get_decimal_point(), 1);

  hpd.shift(-1); // shift right one again
  // result should be 1.2345 again

  EXPECT_EQ(digits[0], uint8_t(1));
  EXPECT_EQ(digits[1], uint8_t(2));
  EXPECT_EQ(digits[2], uint8_t(3));
  EXPECT_EQ(digits[3], uint8_t(4));
  EXPECT_EQ(digits[4], uint8_t(5));
  EXPECT_EQ(hpd.get_num_digits(), 5u);
  EXPECT_EQ(hpd.get_decimal_point(), 1);
}

TEST(LlvmLibcHighPrecisionDecimalTest, MediumShift) {
  __llvm_libc::internal::HighPrecisionDecimal hpd =
      __llvm_libc::internal::HighPrecisionDecimal(".299792458");
  uint8_t *digits = hpd.get_digits();

  hpd.shift(-3); // shift right three, equal to dividing by 8
  // result should be 0.03747405725

  EXPECT_EQ(digits[0], uint8_t(3));
  EXPECT_EQ(digits[1], uint8_t(7));
  EXPECT_EQ(digits[2], uint8_t(4));
  EXPECT_EQ(digits[3], uint8_t(7));
  EXPECT_EQ(digits[4], uint8_t(4));
  EXPECT_EQ(digits[5], uint8_t(0));
  EXPECT_EQ(digits[6], uint8_t(5));
  EXPECT_EQ(digits[7], uint8_t(7));
  EXPECT_EQ(digits[8], uint8_t(2));
  EXPECT_EQ(digits[9], uint8_t(5));
  EXPECT_EQ(hpd.get_num_digits(), 10u);
  EXPECT_EQ(hpd.get_decimal_point(), -1);

  hpd.shift(3); // shift left three, equal to multiplying by 8
  // result should be 0.299792458 again

  EXPECT_EQ(digits[0], uint8_t(2));
  EXPECT_EQ(digits[1], uint8_t(9));
  EXPECT_EQ(digits[2], uint8_t(9));
  EXPECT_EQ(digits[3], uint8_t(7));
  EXPECT_EQ(digits[4], uint8_t(9));
  EXPECT_EQ(digits[5], uint8_t(2));
  EXPECT_EQ(digits[6], uint8_t(4));
  EXPECT_EQ(digits[7], uint8_t(5));
  EXPECT_EQ(digits[8], uint8_t(8));
  EXPECT_EQ(hpd.get_num_digits(), 9u);
  EXPECT_EQ(hpd.get_decimal_point(), 0);
}

TEST(LlvmLibcHighPrecisionDecimalTest, BigShift) {
  __llvm_libc::internal::HighPrecisionDecimal hpd =
      __llvm_libc::internal::HighPrecisionDecimal(".299792458");
  uint8_t *digits = hpd.get_digits();

  hpd.shift(-29); // shift right 29, equal to dividing by 536,870,912
  // result should be 0.0000000005584069676697254180908203125

  EXPECT_EQ(digits[0], uint8_t(5));
  EXPECT_EQ(digits[1], uint8_t(5));
  EXPECT_EQ(digits[2], uint8_t(8));
  EXPECT_EQ(digits[3], uint8_t(4));
  EXPECT_EQ(digits[4], uint8_t(0));
  EXPECT_EQ(digits[5], uint8_t(6));
  EXPECT_EQ(digits[6], uint8_t(9));
  EXPECT_EQ(digits[7], uint8_t(6));
  EXPECT_EQ(digits[8], uint8_t(7));
  EXPECT_EQ(digits[9], uint8_t(6));
  EXPECT_EQ(digits[10], uint8_t(6));
  EXPECT_EQ(digits[11], uint8_t(9));
  EXPECT_EQ(digits[12], uint8_t(7));
  EXPECT_EQ(digits[13], uint8_t(2));
  EXPECT_EQ(digits[14], uint8_t(5));
  EXPECT_EQ(digits[15], uint8_t(4));
  EXPECT_EQ(digits[16], uint8_t(1));
  EXPECT_EQ(digits[17], uint8_t(8));
  EXPECT_EQ(digits[18], uint8_t(0));
  EXPECT_EQ(digits[19], uint8_t(9));
  EXPECT_EQ(digits[20], uint8_t(0));
  EXPECT_EQ(digits[21], uint8_t(8));
  EXPECT_EQ(digits[22], uint8_t(2));
  EXPECT_EQ(digits[23], uint8_t(0));
  EXPECT_EQ(digits[24], uint8_t(3));
  EXPECT_EQ(digits[25], uint8_t(1));
  EXPECT_EQ(digits[26], uint8_t(2));
  EXPECT_EQ(digits[27], uint8_t(5));
  EXPECT_EQ(hpd.get_num_digits(), 28u);
  EXPECT_EQ(hpd.get_decimal_point(), -9);

  hpd.shift(29); // shift left 29, equal to multiplying by 536,870,912
  // result should be 0.299792458 again

  EXPECT_EQ(digits[0], uint8_t(2));
  EXPECT_EQ(digits[1], uint8_t(9));
  EXPECT_EQ(digits[2], uint8_t(9));
  EXPECT_EQ(digits[3], uint8_t(7));
  EXPECT_EQ(digits[4], uint8_t(9));
  EXPECT_EQ(digits[5], uint8_t(2));
  EXPECT_EQ(digits[6], uint8_t(4));
  EXPECT_EQ(digits[7], uint8_t(5));
  EXPECT_EQ(digits[8], uint8_t(8));
  EXPECT_EQ(hpd.get_num_digits(), 9u);
  EXPECT_EQ(hpd.get_decimal_point(), 0);
}

TEST(LlvmLibcHighPrecisionDecimalTest, BigShiftInSteps) {
  __llvm_libc::internal::HighPrecisionDecimal hpd =
      __llvm_libc::internal::HighPrecisionDecimal("1");
  uint8_t *digits = hpd.get_digits();

  hpd.shift(60); // shift left 60, equal to multiplying by
                 // 1152921504606846976.

  EXPECT_EQ(digits[0], uint8_t(1));
  EXPECT_EQ(digits[1], uint8_t(1));
  EXPECT_EQ(digits[2], uint8_t(5));
  EXPECT_EQ(digits[3], uint8_t(2));
  EXPECT_EQ(digits[4], uint8_t(9));
  EXPECT_EQ(digits[5], uint8_t(2));
  EXPECT_EQ(digits[6], uint8_t(1));
  EXPECT_EQ(digits[7], uint8_t(5));
  EXPECT_EQ(digits[8], uint8_t(0));
  EXPECT_EQ(digits[9], uint8_t(4));
  EXPECT_EQ(digits[10], uint8_t(6));
  EXPECT_EQ(digits[11], uint8_t(0));
  EXPECT_EQ(digits[12], uint8_t(6));
  EXPECT_EQ(digits[13], uint8_t(8));
  EXPECT_EQ(digits[14], uint8_t(4));
  EXPECT_EQ(digits[15], uint8_t(6));
  EXPECT_EQ(digits[16], uint8_t(9));
  EXPECT_EQ(digits[17], uint8_t(7));
  EXPECT_EQ(digits[18], uint8_t(6));
  EXPECT_EQ(hpd.get_num_digits(), 19u);
  EXPECT_EQ(hpd.get_decimal_point(), 19);

  hpd.shift(40); // shift left 40, equal to multiplying by
                 // 1099511627776. Result should be 2^100

  EXPECT_EQ(digits[0], uint8_t(1));
  EXPECT_EQ(digits[1], uint8_t(2));
  EXPECT_EQ(digits[2], uint8_t(6));
  EXPECT_EQ(digits[3], uint8_t(7));
  EXPECT_EQ(digits[4], uint8_t(6));
  EXPECT_EQ(digits[5], uint8_t(5));
  EXPECT_EQ(digits[6], uint8_t(0));
  EXPECT_EQ(digits[7], uint8_t(6));
  EXPECT_EQ(digits[8], uint8_t(0));
  EXPECT_EQ(digits[9], uint8_t(0));
  EXPECT_EQ(digits[10], uint8_t(2));
  EXPECT_EQ(digits[11], uint8_t(2));
  EXPECT_EQ(digits[12], uint8_t(8));
  EXPECT_EQ(digits[13], uint8_t(2));
  EXPECT_EQ(digits[14], uint8_t(2));
  EXPECT_EQ(digits[15], uint8_t(9));
  EXPECT_EQ(digits[16], uint8_t(4));
  EXPECT_EQ(digits[17], uint8_t(0));
  EXPECT_EQ(digits[18], uint8_t(1));
  EXPECT_EQ(digits[19], uint8_t(4));
  EXPECT_EQ(digits[20], uint8_t(9));
  EXPECT_EQ(digits[21], uint8_t(6));
  EXPECT_EQ(digits[22], uint8_t(7));
  EXPECT_EQ(digits[23], uint8_t(0));
  EXPECT_EQ(digits[24], uint8_t(3));
  EXPECT_EQ(digits[25], uint8_t(2));
  EXPECT_EQ(digits[26], uint8_t(0));
  EXPECT_EQ(digits[27], uint8_t(5));
  EXPECT_EQ(digits[28], uint8_t(3));
  EXPECT_EQ(digits[29], uint8_t(7));
  EXPECT_EQ(digits[30], uint8_t(6));

  EXPECT_EQ(hpd.get_num_digits(), 31u);
  EXPECT_EQ(hpd.get_decimal_point(), 31);

  hpd.shift(-60); // shift right 60, equal to dividing by
                  // 1152921504606846976. Result should be 2^40

  EXPECT_EQ(digits[0], uint8_t(1));
  EXPECT_EQ(digits[1], uint8_t(0));
  EXPECT_EQ(digits[2], uint8_t(9));
  EXPECT_EQ(digits[3], uint8_t(9));
  EXPECT_EQ(digits[4], uint8_t(5));
  EXPECT_EQ(digits[5], uint8_t(1));
  EXPECT_EQ(digits[6], uint8_t(1));
  EXPECT_EQ(digits[7], uint8_t(6));
  EXPECT_EQ(digits[8], uint8_t(2));
  EXPECT_EQ(digits[9], uint8_t(7));
  EXPECT_EQ(digits[10], uint8_t(7));
  EXPECT_EQ(digits[11], uint8_t(7));
  EXPECT_EQ(digits[12], uint8_t(6));

  EXPECT_EQ(hpd.get_num_digits(), 13u);
  EXPECT_EQ(hpd.get_decimal_point(), 13);

  hpd.shift(-40); // shift right 40, equal to dividing by
                  // 1099511627776. Result should be 1

  EXPECT_EQ(digits[0], uint8_t(1));

  EXPECT_EQ(hpd.get_num_digits(), 1u);
  EXPECT_EQ(hpd.get_decimal_point(), 1);
}

TEST(LlvmLibcHighPrecisionDecimalTest, VeryBigShift) {
  __llvm_libc::internal::HighPrecisionDecimal hpd =
      __llvm_libc::internal::HighPrecisionDecimal("1");
  uint8_t *digits = hpd.get_digits();

  hpd.shift(100); // shift left 100, equal to multiplying by
                  // 1267650600228229401496703205376.
  // result should be 2^100

  EXPECT_EQ(digits[0], uint8_t(1));
  EXPECT_EQ(digits[1], uint8_t(2));
  EXPECT_EQ(digits[2], uint8_t(6));
  EXPECT_EQ(digits[3], uint8_t(7));
  EXPECT_EQ(digits[4], uint8_t(6));
  EXPECT_EQ(digits[5], uint8_t(5));
  EXPECT_EQ(digits[6], uint8_t(0));
  EXPECT_EQ(digits[7], uint8_t(6));
  EXPECT_EQ(digits[8], uint8_t(0));
  EXPECT_EQ(digits[9], uint8_t(0));
  EXPECT_EQ(digits[10], uint8_t(2));
  EXPECT_EQ(digits[11], uint8_t(2));
  EXPECT_EQ(digits[12], uint8_t(8));
  EXPECT_EQ(digits[13], uint8_t(2));
  EXPECT_EQ(digits[14], uint8_t(2));
  EXPECT_EQ(digits[15], uint8_t(9));
  EXPECT_EQ(digits[16], uint8_t(4));
  EXPECT_EQ(digits[17], uint8_t(0));
  EXPECT_EQ(digits[18], uint8_t(1));
  EXPECT_EQ(digits[19], uint8_t(4));
  EXPECT_EQ(digits[20], uint8_t(9));
  EXPECT_EQ(digits[21], uint8_t(6));
  EXPECT_EQ(digits[22], uint8_t(7));
  EXPECT_EQ(digits[23], uint8_t(0));
  EXPECT_EQ(digits[24], uint8_t(3));
  EXPECT_EQ(digits[25], uint8_t(2));
  EXPECT_EQ(digits[26], uint8_t(0));
  EXPECT_EQ(digits[27], uint8_t(5));
  EXPECT_EQ(digits[28], uint8_t(3));
  EXPECT_EQ(digits[29], uint8_t(7));
  EXPECT_EQ(digits[30], uint8_t(6));

  EXPECT_EQ(hpd.get_num_digits(), 31u);
  EXPECT_EQ(hpd.get_decimal_point(), 31);

  hpd.shift(-100); // shift right 100, equal to dividing by
                   // 1267650600228229401496703205376.
  // result should be 1

  EXPECT_EQ(digits[0], uint8_t(1));
  EXPECT_EQ(hpd.get_num_digits(), 1u);
  EXPECT_EQ(hpd.get_decimal_point(), 1);
}

TEST(LlvmLibcHighPrecisionDecimalTest, RoundingTest) {
  __llvm_libc::internal::HighPrecisionDecimal hpd =
      __llvm_libc::internal::HighPrecisionDecimal("1.2345");

  EXPECT_EQ(hpd.round_to_integer_type<uint32_t>(), uint32_t(1));
  EXPECT_EQ(hpd.round_to_integer_type<uint64_t>(), uint64_t(1));
  EXPECT_EQ(hpd.round_to_integer_type<UInt128>(), UInt128(1));

  hpd.shift(1); // shift left 1 to get 2.469 (rounds to 2)

  EXPECT_EQ(hpd.round_to_integer_type<uint32_t>(), uint32_t(2));
  EXPECT_EQ(hpd.round_to_integer_type<uint64_t>(), uint64_t(2));
  EXPECT_EQ(hpd.round_to_integer_type<UInt128>(), UInt128(2));

  hpd.shift(1); // shift left 1 to get 4.938 (rounds to 5)

  EXPECT_EQ(hpd.round_to_integer_type<uint32_t>(), uint32_t(5));
  EXPECT_EQ(hpd.round_to_integer_type<uint64_t>(), uint64_t(5));
  EXPECT_EQ(hpd.round_to_integer_type<UInt128>(), UInt128(5));

  // 2.5 is right between two integers, so we round to even (2)
  hpd = __llvm_libc::internal::HighPrecisionDecimal("2.5");

  EXPECT_EQ(hpd.round_to_integer_type<uint32_t>(), uint32_t(2));
  EXPECT_EQ(hpd.round_to_integer_type<uint64_t>(), uint64_t(2));
  EXPECT_EQ(hpd.round_to_integer_type<UInt128>(), UInt128(2));

  // unless it's marked as having truncated, which means it's actually slightly
  // higher, forcing a round up (3)
  hpd.set_truncated(true);

  EXPECT_EQ(hpd.round_to_integer_type<uint32_t>(), uint32_t(3));
  EXPECT_EQ(hpd.round_to_integer_type<uint64_t>(), uint64_t(3));
  EXPECT_EQ(hpd.round_to_integer_type<UInt128>(), UInt128(3));

  // Check that the larger int types are being handled properly (overflow is not
  // handled, so int types that are too small are ignored for this test.)

  // 1099511627776 = 2^40
  hpd = __llvm_libc::internal::HighPrecisionDecimal("1099511627776");

  EXPECT_EQ(hpd.round_to_integer_type<uint64_t>(), uint64_t(1099511627776));
  EXPECT_EQ(hpd.round_to_integer_type<UInt128>(), UInt128(1099511627776));

  // 1267650600228229401496703205376 = 2^100
  hpd = __llvm_libc::internal::HighPrecisionDecimal(
      "1267650600228229401496703205376");

  UInt128 result = UInt128(1) << 100;

  EXPECT_EQ(hpd.round_to_integer_type<UInt128>(), result);
}
