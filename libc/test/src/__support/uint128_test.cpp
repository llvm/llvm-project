//===-- Unittests for the 128 bit integer class ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/optional.h"
#include "src/__support/UInt.h"

#include "utils/UnitTest/Test.h"

// We want to test __llvm_libc::cpp::UInt<128> explicitly. So, for convenience,
// we use a sugar which does not conflict with the UInt128 type which can
// resolve to __uint128_t if the platform has it.
using LL_UInt128 = __llvm_libc::cpp::UInt<128>;

TEST(LlvmLibcUInt128ClassTest, BasicInit) {
  LL_UInt128 empty;
  LL_UInt128 half_val(12345);
  LL_UInt128 full_val({12345, 67890});
  ASSERT_TRUE(half_val != full_val);
}

TEST(LlvmLibcUInt128ClassTest, AdditionTests) {
  LL_UInt128 val1(12345);
  LL_UInt128 val2(54321);
  LL_UInt128 result1(66666);
  EXPECT_EQ(val1 + val2, result1);
  EXPECT_EQ((val1 + val2), (val2 + val1)); // addition is commutative

  // Test overflow
  LL_UInt128 val3({0xf000000000000001, 0});
  LL_UInt128 val4({0x100000000000000f, 0});
  LL_UInt128 result2({0x10, 0x1});
  EXPECT_EQ(val3 + val4, result2);
  EXPECT_EQ(val3 + val4, val4 + val3);

  // Test overflow
  LL_UInt128 val5({0x0123456789abcdef, 0xfedcba9876543210});
  LL_UInt128 val6({0x1111222233334444, 0xaaaabbbbccccdddd});
  LL_UInt128 result3({0x12346789bcdf1233, 0xa987765443210fed});
  EXPECT_EQ(val5 + val6, result3);
  EXPECT_EQ(val5 + val6, val6 + val5);
}

TEST(LlvmLibcUInt128ClassTest, SubtractionTests) {
  LL_UInt128 val1(12345);
  LL_UInt128 val2(54321);
  LL_UInt128 result1({0xffffffffffff5c08, 0xffffffffffffffff});
  LL_UInt128 result2(0xa3f8);
  EXPECT_EQ(val1 - val2, result1);
  EXPECT_EQ(val1, val2 + result1);
  EXPECT_EQ(val2 - val1, result2);
  EXPECT_EQ(val2, val1 + result2);

  LL_UInt128 val3({0xf000000000000001, 0});
  LL_UInt128 val4({0x100000000000000f, 0});
  LL_UInt128 result3(0xdffffffffffffff2);
  LL_UInt128 result4({0x200000000000000e, 0xffffffffffffffff});
  EXPECT_EQ(val3 - val4, result3);
  EXPECT_EQ(val3, val4 + result3);
  EXPECT_EQ(val4 - val3, result4);
  EXPECT_EQ(val4, val3 + result4);

  LL_UInt128 val5({0x0123456789abcdef, 0xfedcba9876543210});
  LL_UInt128 val6({0x1111222233334444, 0xaaaabbbbccccdddd});
  LL_UInt128 result5({0xf0122345567889ab, 0x5431fedca9875432});
  LL_UInt128 result6({0x0feddcbaa9877655, 0xabce01235678abcd});
  EXPECT_EQ(val5 - val6, result5);
  EXPECT_EQ(val5, val6 + result5);
  EXPECT_EQ(val6 - val5, result6);
  EXPECT_EQ(val6, val5 + result6);
}

TEST(LlvmLibcUInt128ClassTest, MultiplicationTests) {
  LL_UInt128 val1({5, 0});
  LL_UInt128 val2({10, 0});
  LL_UInt128 result1({50, 0});
  EXPECT_EQ((val1 * val2), result1);
  EXPECT_EQ((val1 * val2), (val2 * val1)); // multiplication is commutative

  // Check that the multiplication works accross the whole number
  LL_UInt128 val3({0xf, 0});
  LL_UInt128 val4({0x1111111111111111, 0x1111111111111111});
  LL_UInt128 result2({0xffffffffffffffff, 0xffffffffffffffff});
  EXPECT_EQ((val3 * val4), result2);
  EXPECT_EQ((val3 * val4), (val4 * val3));

  // Check that multiplication doesn't reorder the bits.
  LL_UInt128 val5({2, 0});
  LL_UInt128 val6({0x1357024675316420, 0x0123456776543210});
  LL_UInt128 result3({0x26ae048cea62c840, 0x02468aceeca86420});

  EXPECT_EQ((val5 * val6), result3);
  EXPECT_EQ((val5 * val6), (val6 * val5));

  // Make sure that multiplication handles overflow correctly.
  LL_UInt128 val7(2);
  LL_UInt128 val8({0x8000800080008000, 0x8000800080008000});
  LL_UInt128 result4({0x0001000100010000, 0x0001000100010001});
  EXPECT_EQ((val7 * val8), result4);
  EXPECT_EQ((val7 * val8), (val8 * val7));

  // val9 is the 128 bit mantissa of 1e60 as a float, val10 is the mantissa for
  // 1e-60. They almost cancel on the high bits, but the result we're looking
  // for is just the low bits. The full result would be
  // 0x7fffffffffffffffffffffffffffffff3a4f32d17f40d08f917cf11d1e039c50
  LL_UInt128 val9({0x01D762422C946590, 0x9F4F2726179A2245});
  LL_UInt128 val10({0x3792F412CB06794D, 0xCDB02555653131B6});
  LL_UInt128 result5({0x917cf11d1e039c50, 0x3a4f32d17f40d08f});
  EXPECT_EQ((val9 * val10), result5);
  EXPECT_EQ((val9 * val10), (val10 * val9));
}

TEST(LlvmLibcUInt128ClassTest, DivisionTests) {
  LL_UInt128 val1({10, 0});
  LL_UInt128 val2({5, 0});
  LL_UInt128 result1({2, 0});
  EXPECT_EQ((val1 / val2), result1);
  EXPECT_EQ((val1 / result1), val2);

  // Check that the division works accross the whole number
  LL_UInt128 val3({0xffffffffffffffff, 0xffffffffffffffff});
  LL_UInt128 val4({0xf, 0});
  LL_UInt128 result2({0x1111111111111111, 0x1111111111111111});
  EXPECT_EQ((val3 / val4), result2);
  EXPECT_EQ((val3 / result2), val4);

  // Check that division doesn't reorder the bits.
  LL_UInt128 val5({0x26ae048cea62c840, 0x02468aceeca86420});
  LL_UInt128 val6({2, 0});
  LL_UInt128 result3({0x1357024675316420, 0x0123456776543210});
  EXPECT_EQ((val5 / val6), result3);
  EXPECT_EQ((val5 / result3), val6);

  // Make sure that division handles inexact results correctly.
  LL_UInt128 val7({1001, 0});
  LL_UInt128 val8({10, 0});
  LL_UInt128 result4({100, 0});
  EXPECT_EQ((val7 / val8), result4);
  EXPECT_EQ((val7 / result4), val8);

  // Make sure that division handles divisors of one correctly.
  LL_UInt128 val9({0x1234567812345678, 0x9abcdef09abcdef0});
  LL_UInt128 val10({1, 0});
  LL_UInt128 result5({0x1234567812345678, 0x9abcdef09abcdef0});
  EXPECT_EQ((val9 / val10), result5);
  EXPECT_EQ((val9 / result5), val10);

  // Make sure that division handles results of slightly more than 1 correctly.
  LL_UInt128 val11({1050, 0});
  LL_UInt128 val12({1030, 0});
  LL_UInt128 result6({1, 0});
  EXPECT_EQ((val11 / val12), result6);

  // Make sure that division handles dividing by zero correctly.
  LL_UInt128 val13({1234, 0});
  LL_UInt128 val14({0, 0});
  EXPECT_FALSE(val13.div(val14).has_value());
}

TEST(LlvmLibcUInt128ClassTest, ModuloTests) {
  LL_UInt128 val1({10, 0});
  LL_UInt128 val2({5, 0});
  LL_UInt128 result1({0, 0});
  EXPECT_EQ((val1 % val2), result1);

  LL_UInt128 val3({101, 0});
  LL_UInt128 val4({10, 0});
  LL_UInt128 result2({1, 0});
  EXPECT_EQ((val3 % val4), result2);

  LL_UInt128 val5({10000001, 0});
  LL_UInt128 val6({10, 0});
  LL_UInt128 result3({1, 0});
  EXPECT_EQ((val5 % val6), result3);

  LL_UInt128 val7({12345, 10});
  LL_UInt128 val8({0, 1});
  LL_UInt128 result4({12345, 0});
  EXPECT_EQ((val7 % val8), result4);

  LL_UInt128 val9({12345, 10});
  LL_UInt128 val10({0, 11});
  LL_UInt128 result5({12345, 10});
  EXPECT_EQ((val9 % val10), result5);

  LL_UInt128 val11({10, 10});
  LL_UInt128 val12({10, 10});
  LL_UInt128 result6({0, 0});
  EXPECT_EQ((val11 % val12), result6);

  LL_UInt128 val13({12345, 0});
  LL_UInt128 val14({1, 0});
  LL_UInt128 result7({0, 0});
  EXPECT_EQ((val13 % val14), result7);

  LL_UInt128 val15({0xffffffffffffffff, 0xffffffffffffffff});
  LL_UInt128 val16({0x1111111111111111, 0x111111111111111});
  LL_UInt128 result8({0xf, 0});
  EXPECT_EQ((val15 % val16), result8);

  LL_UInt128 val17({5076944270305263619, 54210108624}); // (10 ^ 30) + 3
  LL_UInt128 val18({10, 0});
  LL_UInt128 result9({3, 0});
  EXPECT_EQ((val17 % val18), result9);
}

TEST(LlvmLibcUInt128ClassTest, PowerTests) {
  LL_UInt128 val1({10, 0});
  val1.pow_n(30);
  LL_UInt128 result1({5076944270305263616, 54210108624}); // (10 ^ 30)
  EXPECT_EQ(val1, result1);

  LL_UInt128 val2({1, 0});
  val2.pow_n(10);
  LL_UInt128 result2({1, 0});
  EXPECT_EQ(val2, result2);

  LL_UInt128 val3({0, 0});
  val3.pow_n(10);
  LL_UInt128 result3({0, 0});
  EXPECT_EQ(val3, result3);

  LL_UInt128 val4({10, 0});
  val4.pow_n(0);
  LL_UInt128 result4({1, 0});
  EXPECT_EQ(val4, result4);

  // Test zero to the zero. Currently it returns 1, since that's the easiest
  // result.
  LL_UInt128 val5({0, 0});
  val5.pow_n(0);
  LL_UInt128 result5({1, 0});
  EXPECT_EQ(val5, result5);

  // Test a number that overflows. 100 ^ 20 is larger than 2 ^ 128.
  LL_UInt128 val6({100, 0});
  val6.pow_n(20);
  LL_UInt128 result6({0xb9f5610000000000, 0x6329f1c35ca4bfab});
  EXPECT_EQ(val6, result6);

  // Test that both halves of the number are being used.
  LL_UInt128 val7({1, 1});
  val7.pow_n(2);
  LL_UInt128 result7({1, 2});
  EXPECT_EQ(val7, result7);

  LL_UInt128 val_pow_two;
  LL_UInt128 result_pow_two;
  for (size_t i = 0; i < 128; ++i) {
    val_pow_two = 2;
    val_pow_two.pow_n(i);
    result_pow_two = 1;
    result_pow_two = result_pow_two << i;
    EXPECT_EQ(val_pow_two, result_pow_two);
  }
}

TEST(LlvmLibcUInt128ClassTest, ShiftLeftTests) {
  LL_UInt128 val1(0x0123456789abcdef);
  LL_UInt128 result1(0x123456789abcdef0);
  EXPECT_EQ((val1 << 4), result1);

  LL_UInt128 val2({0x13579bdf02468ace, 0x123456789abcdef0});
  LL_UInt128 result2({0x02468ace00000000, 0x9abcdef013579bdf});
  EXPECT_EQ((val2 << 32), result2);
  LL_UInt128 val22 = val2;
  val22 <<= 32;
  EXPECT_EQ(val22, result2);

  LL_UInt128 result3({0, 0x13579bdf02468ace});
  EXPECT_EQ((val2 << 64), result3);

  LL_UInt128 result4({0, 0x02468ace00000000});
  EXPECT_EQ((val2 << 96), result4);

  LL_UInt128 result5({0, 0x2468ace000000000});
  EXPECT_EQ((val2 << 100), result5);

  LL_UInt128 result6({0, 0});
  EXPECT_EQ((val2 << 128), result6);
  EXPECT_EQ((val2 << 256), result6);
}

TEST(LlvmLibcUInt128ClassTest, ShiftRightTests) {
  LL_UInt128 val1(0x0123456789abcdef);
  LL_UInt128 result1(0x00123456789abcde);
  EXPECT_EQ((val1 >> 4), result1);

  LL_UInt128 val2({0x13579bdf02468ace, 0x123456789abcdef0});
  LL_UInt128 result2({0x9abcdef013579bdf, 0x0000000012345678});
  EXPECT_EQ((val2 >> 32), result2);
  LL_UInt128 val22 = val2;
  val22 >>= 32;
  EXPECT_EQ(val22, result2);

  LL_UInt128 result3({0x123456789abcdef0, 0});
  EXPECT_EQ((val2 >> 64), result3);

  LL_UInt128 result4({0x0000000012345678, 0});
  EXPECT_EQ((val2 >> 96), result4);

  LL_UInt128 result5({0x0000000001234567, 0});
  EXPECT_EQ((val2 >> 100), result5);

  LL_UInt128 result6({0, 0});
  EXPECT_EQ((val2 >> 128), result6);
  EXPECT_EQ((val2 >> 256), result6);
}

TEST(LlvmLibcUInt128ClassTest, AndTests) {
  LL_UInt128 base({0xffff00000000ffff, 0xffffffff00000000});
  LL_UInt128 val128({0xf0f0f0f00f0f0f0f, 0xff00ff0000ff00ff});
  uint64_t val64 = 0xf0f0f0f00f0f0f0f;
  int val32 = 0x0f0f0f0f;
  LL_UInt128 result128({0xf0f0000000000f0f, 0xff00ff0000000000});
  LL_UInt128 result64(0xf0f0000000000f0f);
  LL_UInt128 result32(0x00000f0f);
  EXPECT_EQ((base & val128), result128);
  EXPECT_EQ((base & val64), result64);
  EXPECT_EQ((base & val32), result32);
}

TEST(LlvmLibcUInt128ClassTest, OrTests) {
  LL_UInt128 base({0xffff00000000ffff, 0xffffffff00000000});
  LL_UInt128 val128({0xf0f0f0f00f0f0f0f, 0xff00ff0000ff00ff});
  uint64_t val64 = 0xf0f0f0f00f0f0f0f;
  int val32 = 0x0f0f0f0f;
  LL_UInt128 result128({0xfffff0f00f0fffff, 0xffffffff00ff00ff});
  LL_UInt128 result64({0xfffff0f00f0fffff, 0xffffffff00000000});
  LL_UInt128 result32({0xffff00000f0fffff, 0xffffffff00000000});
  EXPECT_EQ((base | val128), result128);
  EXPECT_EQ((base | val64), result64);
  EXPECT_EQ((base | val32), result32);
}

TEST(LlvmLibcUInt128ClassTest, CompoundAssignments) {
  LL_UInt128 x({0xffff00000000ffff, 0xffffffff00000000});
  LL_UInt128 b({0xf0f0f0f00f0f0f0f, 0xff00ff0000ff00ff});

  LL_UInt128 a = x;
  a |= b;
  LL_UInt128 or_result({0xfffff0f00f0fffff, 0xffffffff00ff00ff});
  EXPECT_EQ(a, or_result);

  a = x;
  a &= b;
  LL_UInt128 and_result({0xf0f0000000000f0f, 0xff00ff0000000000});
  EXPECT_EQ(a, and_result);

  a = x;
  a ^= b;
  LL_UInt128 xor_result({0x0f0ff0f00f0ff0f0, 0x00ff00ff00ff00ff});
  EXPECT_EQ(a, xor_result);

  a = LL_UInt128(uint64_t(0x0123456789abcdef));
  LL_UInt128 shift_left_result(uint64_t(0x123456789abcdef0));
  a <<= 4;
  EXPECT_EQ(a, shift_left_result);

  a = LL_UInt128(uint64_t(0x123456789abcdef1));
  LL_UInt128 shift_right_result(uint64_t(0x0123456789abcdef));
  a >>= 4;
  EXPECT_EQ(a, shift_right_result);

  a = LL_UInt128({0xf000000000000001, 0});
  b = LL_UInt128({0x100000000000000f, 0});
  LL_UInt128 add_result({0x10, 0x1});
  a += b;
  EXPECT_EQ(a, add_result);

  a = LL_UInt128({0xf, 0});
  b = LL_UInt128({0x1111111111111111, 0x1111111111111111});
  LL_UInt128 mul_result({0xffffffffffffffff, 0xffffffffffffffff});
  a *= b;
  EXPECT_EQ(a, mul_result);
}

TEST(LlvmLibcUInt128ClassTest, UnaryPredecrement) {
  LL_UInt128 a = LL_UInt128({0x1111111111111111, 0x1111111111111111});
  ++a;
  EXPECT_EQ(a, LL_UInt128({0x1111111111111112, 0x1111111111111111}));

  a = LL_UInt128({0xffffffffffffffff, 0x0});
  ++a;
  EXPECT_EQ(a, LL_UInt128({0x0, 0x1}));

  a = LL_UInt128({0xffffffffffffffff, 0xffffffffffffffff});
  ++a;
  EXPECT_EQ(a, LL_UInt128({0x0, 0x0}));
}

TEST(LlvmLibcUInt128ClassTest, EqualsTests) {
  LL_UInt128 a1({0xffffffff00000000, 0xffff00000000ffff});
  LL_UInt128 a2({0xffffffff00000000, 0xffff00000000ffff});
  LL_UInt128 b({0xff00ff0000ff00ff, 0xf0f0f0f00f0f0f0f});
  LL_UInt128 a_reversed({0xffff00000000ffff, 0xffffffff00000000});
  LL_UInt128 a_upper(0xffff00000000ffff);
  LL_UInt128 a_lower(0xffffffff00000000);
  ASSERT_TRUE(a1 == a1);
  ASSERT_TRUE(a1 == a2);
  ASSERT_FALSE(a1 == b);
  ASSERT_FALSE(a1 == a_reversed);
  ASSERT_FALSE(a1 == a_lower);
  ASSERT_FALSE(a1 == a_upper);
  ASSERT_TRUE(a_lower != a_upper);
}

TEST(LlvmLibcUInt128ClassTest, ComparisonTests) {
  LL_UInt128 a({0xffffffff00000000, 0xffff00000000ffff});
  LL_UInt128 b({0xff00ff0000ff00ff, 0xf0f0f0f00f0f0f0f});
  EXPECT_GT(a, b);
  EXPECT_GE(a, b);
  EXPECT_LT(b, a);
  EXPECT_LE(b, a);

  LL_UInt128 x(0xffffffff00000000);
  LL_UInt128 y(0x00000000ffffffff);
  EXPECT_GT(x, y);
  EXPECT_GE(x, y);
  EXPECT_LT(y, x);
  EXPECT_LE(y, x);

  EXPECT_LE(a, a);
  EXPECT_GE(a, a);
}
