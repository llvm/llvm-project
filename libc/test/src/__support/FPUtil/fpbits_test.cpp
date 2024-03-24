//===-- Unittests for the FPBits class ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/fpbits_str.h"
#include "src/__support/integer_literals.h"
#include "src/__support/sign.h" // Sign
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::FPBits;
using LIBC_NAMESPACE::fputil::FPType;
using LIBC_NAMESPACE::fputil::internal::FPRep;

using LIBC_NAMESPACE::operator""_u16;
using LIBC_NAMESPACE::operator""_u32;
using LIBC_NAMESPACE::operator""_u64;
using LIBC_NAMESPACE::operator""_u128;

TEST(LlvmLibcFPBitsTest, FPType_IEEE754_Binary16) {
  using Rep = FPRep<FPType::IEEE754_Binary16>;
  using u16 = typename Rep::StorageType;

  EXPECT_EQ(0b0'00000'0000000000_u16, u16(Rep::zero()));
  EXPECT_EQ(0b0'01111'0000000000_u16, u16(Rep::one()));
  EXPECT_EQ(0b0'00000'0000000001_u16, u16(Rep::min_subnormal()));
  EXPECT_EQ(0b0'00000'1111111111_u16, u16(Rep::max_subnormal()));
  EXPECT_EQ(0b0'00001'0000000000_u16, u16(Rep::min_normal()));
  EXPECT_EQ(0b0'11110'1111111111_u16, u16(Rep::max_normal()));
  EXPECT_EQ(0b0'11111'0000000000_u16, u16(Rep::inf()));
  EXPECT_EQ(0b0'11111'0100000000_u16, u16(Rep::signaling_nan()));
  EXPECT_EQ(0b0'11111'1000000000_u16, u16(Rep::quiet_nan()));
}

TEST(LlvmLibcFPBitsTest, FPType_IEEE754_Binary32) {
  using Rep = FPRep<FPType::IEEE754_Binary32>;
  using u32 = typename Rep::StorageType;

  EXPECT_EQ(0b0'00000000'00000000000000000000000_u32, u32(Rep::zero()));
  EXPECT_EQ(0b0'01111111'00000000000000000000000_u32, u32(Rep::one()));
  EXPECT_EQ(0b0'00000000'00000000000000000000001_u32,
            u32(Rep::min_subnormal()));
  EXPECT_EQ(0b0'00000000'11111111111111111111111_u32,
            u32(Rep::max_subnormal()));
  EXPECT_EQ(0b0'00000001'00000000000000000000000_u32, u32(Rep::min_normal()));
  EXPECT_EQ(0b0'11111110'11111111111111111111111_u32, u32(Rep::max_normal()));
  EXPECT_EQ(0b0'11111111'00000000000000000000000_u32, u32(Rep::inf()));
  EXPECT_EQ(0b0'11111111'01000000000000000000000_u32,
            u32(Rep::signaling_nan()));
  EXPECT_EQ(0b0'11111111'10000000000000000000000_u32, u32(Rep::quiet_nan()));
}

TEST(LlvmLibcFPBitsTest, FPType_IEEE754_Binary64) {
  using Rep = FPRep<FPType::IEEE754_Binary64>;
  using u64 = typename Rep::StorageType;

  EXPECT_EQ(
      0b0'00000000000'0000000000000000000000000000000000000000000000000000_u64,
      u64(Rep::zero()));
  EXPECT_EQ(
      0b0'01111111111'0000000000000000000000000000000000000000000000000000_u64,
      u64(Rep::one()));
  EXPECT_EQ(
      0b0'00000000000'0000000000000000000000000000000000000000000000000001_u64,
      u64(Rep::min_subnormal()));
  EXPECT_EQ(
      0b0'00000000000'1111111111111111111111111111111111111111111111111111_u64,
      u64(Rep::max_subnormal()));
  EXPECT_EQ(
      0b0'00000000001'0000000000000000000000000000000000000000000000000000_u64,
      u64(Rep::min_normal()));
  EXPECT_EQ(
      0b0'11111111110'1111111111111111111111111111111111111111111111111111_u64,
      u64(Rep::max_normal()));
  EXPECT_EQ(
      0b0'11111111111'0000000000000000000000000000000000000000000000000000_u64,
      u64(Rep::inf()));
  EXPECT_EQ(
      0b0'11111111111'0100000000000000000000000000000000000000000000000000_u64,
      u64(Rep::signaling_nan()));
  EXPECT_EQ(
      0b0'11111111111'1000000000000000000000000000000000000000000000000000_u64,
      u64(Rep::quiet_nan()));
}

TEST(LlvmLibcFPBitsTest, FPType_IEEE754_Binary128) {
  using Rep = FPRep<FPType::IEEE754_Binary128>;

  EXPECT_EQ(
      0b0'000000000000000'0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::zero()));
  EXPECT_EQ(
      0b0'011111111111111'0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::one()));
  EXPECT_EQ(
      0b0'000000000000000'0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001_u128,
      UInt128(Rep::min_subnormal()));
  EXPECT_EQ(
      0b0'000000000000000'1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111_u128,
      UInt128(Rep::max_subnormal()));
  EXPECT_EQ(
      0b0'000000000000001'0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::min_normal()));
  EXPECT_EQ(
      0b0'111111111111110'1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111_u128,
      UInt128(Rep::max_normal()));
  EXPECT_EQ(
      0b0'111111111111111'0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::inf()));
  EXPECT_EQ(
      0b0'111111111111111'0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::signaling_nan()));
  EXPECT_EQ(
      0b0'111111111111111'1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::quiet_nan()));
}

TEST(LlvmLibcFPBitsTest, FPType_X86_Binary80) {
  using Rep = FPRep<FPType::X86_Binary80>;

  EXPECT_EQ(
      0b0'0000000000000000000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::zero()));
  EXPECT_EQ(
      0b0'0111111111111111000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::one()));
  EXPECT_EQ(
      0b0'0000000000000000000000000000000000000000000000000000000000000000000000000000001_u128,
      UInt128(Rep::min_subnormal()));
  EXPECT_EQ(
      0b0'0000000000000000111111111111111111111111111111111111111111111111111111111111111_u128,
      UInt128(Rep::max_subnormal()));
  EXPECT_EQ(
      0b0'0000000000000011000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::min_normal()));
  EXPECT_EQ(
      0b0'1111111111111101111111111111111111111111111111111111111111111111111111111111111_u128,
      UInt128(Rep::max_normal()));
  EXPECT_EQ(
      0b0'1111111111111111000000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::inf()));
  EXPECT_EQ(
      0b0'1111111111111111010000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::signaling_nan()));
  EXPECT_EQ(
      0b0'1111111111111111100000000000000000000000000000000000000000000000000000000000000_u128,
      UInt128(Rep::quiet_nan()));
}

TEST(LlvmLibcFPBitsTest, FPType_X86_Binary80_IsNan) {
  using Rep = FPRep<FPType::X86_Binary80>;

  EXPECT_TRUE( // NAN : Pseudo-Infinity
      Rep(0b0'111111111111111'0000000000000000000000000000000000000000000000000000000000000000_u128)
          .is_nan());
  EXPECT_TRUE( // NAN : Pseudo Not a Number
      Rep(0b0'111111111111111'0000000000000000000000000000000000000000000000000000000000000001_u128)
          .is_nan());
  EXPECT_TRUE( // NAN : Pseudo Not a Number
      Rep(0b0'111111111111111'0100000000000000000000000000000000000000000000000000000000000000_u128)
          .is_nan());
  EXPECT_TRUE( // NAN : Signalling Not a Number
      Rep(0b0'111111111111111'1000000000000000000000000000000000000000000000000000000000000001_u128)
          .is_nan());
  EXPECT_TRUE( // NAN : Floating-point Indefinite
      Rep(0b0'111111111111111'1100000000000000000000000000000000000000000000000000000000000000_u128)
          .is_nan());
  EXPECT_TRUE( // NAN : Quiet Not a Number
      Rep(0b0'111111111111111'1100000000000000000000000000000000000000000000000000000000000001_u128)
          .is_nan());
  EXPECT_TRUE( // NAN : Unnormal
      Rep(0b0'111111111111110'0000000000000000000000000000000000000000000000000000000000000000_u128)
          .is_nan());
  EXPECT_FALSE( // Zero
      Rep(0b0'000000000000000'0000000000000000000000000000000000000000000000000000000000000000_u128)
          .is_nan());
  EXPECT_FALSE( // Subnormal
      Rep(0b0'000000000000000'0000000000000000000000000000000000000000000000000000000000000001_u128)
          .is_nan());
  EXPECT_FALSE( // Pseudo Denormal
      Rep(0b0'000000000000000'1000000000000000000000000000000000000000000000000000000000000001_u128)
          .is_nan());
  EXPECT_FALSE( // Infinity
      Rep(0b0'111111111111111'1000000000000000000000000000000000000000000000000000000000000000_u128)
          .is_nan());
  EXPECT_FALSE( // Normalized
      Rep(0b0'111111111111110'1000000000000000000000000000000000000000000000000000000000000000_u128)
          .is_nan());
}

enum class FP {
  ZERO,
  MIN_SUBNORMAL,
  MAX_SUBNORMAL,
  MIN_NORMAL,
  ONE,
  MAX_NORMAL,
  INF,
  SIGNALING_NAN,
  QUIET_NAN
};

constexpr FP all_fp_values[] = {
    FP::ZERO,       FP::MIN_SUBNORMAL, FP::MAX_SUBNORMAL,
    FP::MIN_NORMAL, FP::ONE,           FP::MAX_NORMAL,
    FP::INF,        FP::SIGNALING_NAN, FP::QUIET_NAN,
};

constexpr Sign all_signs[] = {Sign::POS, Sign::NEG};

using FPTypes =
    LIBC_NAMESPACE::testing::TypeList<FPRep<FPType::IEEE754_Binary16>,  //
                                      FPRep<FPType::IEEE754_Binary32>,  //
                                      FPRep<FPType::IEEE754_Binary64>,  //
                                      FPRep<FPType::IEEE754_Binary128>, //
                                      FPRep<FPType::X86_Binary80>       //
                                      >;

template <typename T> constexpr auto make(Sign sign, FP fp) {
  switch (fp) {
  case FP::ZERO:
    return T::zero(sign);
  case FP::MIN_SUBNORMAL:
    return T::min_subnormal(sign);
  case FP::MAX_SUBNORMAL:
    return T::max_subnormal(sign);
  case FP::MIN_NORMAL:
    return T::min_normal(sign);
  case FP::ONE:
    return T::one(sign);
  case FP::MAX_NORMAL:
    return T::max_normal(sign);
  case FP::INF:
    return T::inf(sign);
  case FP::SIGNALING_NAN:
    return T::signaling_nan(sign);
  case FP::QUIET_NAN:
    return T::quiet_nan(sign);
  }
  __builtin_unreachable();
}

// Tests all properties for all types of float.
TYPED_TEST(LlvmLibcFPBitsTest, Properties, FPTypes) {
  for (Sign sign : all_signs) {
    for (FP fp : all_fp_values) {
      const T value = make<T>(sign, fp);
      // is_zero
      ASSERT_EQ(value.is_zero(), fp == FP::ZERO);
      // is_inf_or_nan
      ASSERT_EQ(value.is_inf_or_nan(), fp == FP::INF ||
                                           fp == FP::SIGNALING_NAN ||
                                           fp == FP::QUIET_NAN);
      // is_finite
      ASSERT_EQ(value.is_finite(), fp != FP::INF && fp != FP::SIGNALING_NAN &&
                                       fp != FP::QUIET_NAN);
      // is_inf
      ASSERT_EQ(value.is_inf(), fp == FP::INF);
      // is_nan
      ASSERT_EQ(value.is_nan(), fp == FP::SIGNALING_NAN || fp == FP::QUIET_NAN);
      // is_normal
      ASSERT_EQ(value.is_normal(),
                fp == FP::MIN_NORMAL || fp == FP::ONE || fp == FP::MAX_NORMAL);
      // is_quiet_nan
      ASSERT_EQ(value.is_quiet_nan(), fp == FP::QUIET_NAN);
      // is_signaling_nan
      ASSERT_EQ(value.is_signaling_nan(), fp == FP::SIGNALING_NAN);
      // is_subnormal
      ASSERT_EQ(value.is_subnormal(), fp == FP::ZERO ||
                                          fp == FP::MIN_SUBNORMAL ||
                                          fp == FP::MAX_SUBNORMAL);
      // is_pos
      ASSERT_EQ(value.is_pos(), sign == Sign::POS);
      ASSERT_EQ(value.sign().is_pos(), sign == Sign::POS);
      // is_neg
      ASSERT_EQ(value.is_neg(), sign == Sign::NEG);
      ASSERT_EQ(value.sign().is_neg(), sign == Sign::NEG);
    }
  }
}

#define ASSERT_SAME_REP(A, B) ASSERT_EQ(A.uintval(), B.uintval());

TYPED_TEST(LlvmLibcFPBitsTest, NextTowardInf, FPTypes) {
  struct {
    FP before, after;
  } TEST_CASES[] = {
      {FP::ZERO, FP::MIN_SUBNORMAL},          //
      {FP::MAX_SUBNORMAL, FP::MIN_NORMAL},    //
      {FP::MAX_NORMAL, FP::INF},              //
      {FP::INF, FP::INF},                     //
      {FP::QUIET_NAN, FP::QUIET_NAN},         //
      {FP::SIGNALING_NAN, FP::SIGNALING_NAN}, //
  };
  for (Sign sign : all_signs) {
    for (auto tc : TEST_CASES) {
      T val = make<T>(sign, tc.before);
      ASSERT_SAME_REP(val.next_toward_inf(), make<T>(sign, tc.after));
    }
  }
}

TYPED_TEST(LlvmLibcFPBitsTest, NumberConstruction, FPTypes) {
  using LIBC_NAMESPACE::cpp::countl_zero;
  using LIBC_NAMESPACE::cpp::countr_zero;
  using Number = typename T::Number;

  // When using get_number() the significand is transfered as-is and the
  // exponent is adjusted to reflect the extra precision (now the significand
  // uses (STORAGE_LEN - 1) bits instead of FRACTION_LEN bits).

  // e.g., with IEEE754_Binary16
  // 1.0 in IEEE754_Binary16 : 0b0011110000000000
  //                             SEEEEEMMMMMMMMMM
  // number's significand    : 0b0000010000000000
  // EXTRA_PRECISION         :   ^^^^^
  // number's exponent       : EXTRA_PRECISION

  const T one = T::one();

  const Number num = one.get_number();

  // "num" and "one" have the same sign.
  ASSERT_EQ(num.sign.is_pos(), one.is_pos());

  // For 'one', the leading one of the significant is at position FRACTION_LEN.
  // So we have FRACTION_LEN zeroes after it.
  ASSERT_EQ(countr_zero(num.significand), T::FRACTION_LEN);

  // The exponent is increased by EXTRA_PRECISION.
  // Since the exponent for 'one' is '0' the number's exponent is just
  // EXTRA_PRECISION.
  ASSERT_EQ(num.exponent, Number::EXTRA_PRECISION);

  // Because the significant is now stored in 'StorageType' we have extra
  // precisions bits available at the left of the leading one.
  ASSERT_GT(Number::EXTRA_PRECISION, 0);
  ASSERT_EQ(countl_zero(num.significand), Number::EXTRA_PRECISION);

  // In maximized precision form, the leading one is moved at StorageType's MSB.
  // number's significand    : 0b1000000000000000
  // number's exponent       : 0
  const Number max_precision = one.get_number().maximize_precision();
  ASSERT_TRUE(max_precision.sign.is_pos());
  // The leading bit is now in the MSB of the storage.
  ASSERT_EQ(countl_zero(max_precision.significand), 0);
  ASSERT_EQ(max_precision.exponent, 0);

  // In minimized precision form, the leading one is moved at StorageType's LSB.
  // number's significand    : 0b0000000000000001
  // number's exponent       : FRACTION_LEN + EXTRA_PRECISION
  const Number min_precision = one.get_number().minimize_precision();
  ASSERT_TRUE(min_precision.sign.is_pos());
  // The leading bit is now in the MSB of the storage.
  ASSERT_EQ(countr_zero(min_precision.significand), 0);
  ASSERT_EQ(min_precision.exponent, T::FRACTION_LEN + Number::EXTRA_PRECISION);
}

#define ASSERT_MATERIALIZE_AS(NUMBER, ROUNDING, PRECISION, REP)                \
  ASSERT_SAME_REP(NUMBER.materialize(ROUNDING, PRECISION), REP)

// For all 'FPType' and all finite 'FP' values, we check that we can convert the
// 'FPRep' to a 'Number' and back to the original 'FPRep' without loss.
// We also check that changing the scale of the intermediary 'Number' has no
// effect.
TYPED_TEST(LlvmLibcFPBitsTest, NumberBackAndForth, FPTypes) {
  // using StorageType = typename T::StorageType;
  using Number = typename T::Number;
  for (Sign sign : all_signs) {
    for (FP fp : all_fp_values) {
      const T rep = make<T>(sign, fp);
      if (!rep.is_finite())
        continue;
      // We test numbers at different scales.
      // Note: changing scale changes the internal representation but not the
      // Number's value.
      const Number scaled_numbers[] = {
          rep.get_number(),
          rep.get_number().maximize_precision(),
          rep.get_number().minimize_precision(),
      };
      for (const Number &num : scaled_numbers) {
        // When numbers are exact (i.e., not truncated) they should materialize
        // back exactly whatever the rounding mode.
        ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::EXACT, rep);
        ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::EXACT, rep);
        ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::EXACT, rep);
      }
    }
  }
}

// Here we test materialization of a 'Number' back to an 'FPRep' with the
// 'TOWARDZERO' rounding mode. This rounding mode corresponds to C++ cast
// semantics and simply discards the extra precision.
// That is, whatever the values of the extra bits, 'Number' will materialize
// back as 'FPRep' exactly.
TYPED_TEST(LlvmLibcFPBitsTest, NumberRoundTowardZero, FPTypes) {
  using StorageType = typename T::StorageType;
  using Number = typename T::Number;
  constexpr auto set_last_bits = [](StorageType value, int bits) {
    return value | ((StorageType(1) << bits) - StorageType(1));
  };
  for (Sign sign : all_signs) {
    for (FP fp : all_fp_values) {
      const T rep = make<T>(sign, fp);
      if (!rep.is_finite())
        continue;
      // Number with extra precision bits.
      Number num = rep.get_number().maximize_precision();
      const int extra_bits = Number::EXTRA_PRECISION + rep.is_subnormal();

      // Exact number converts back to rep.
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::EXACT, rep);
      // Non-exact numbers converts back to rep.
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::TRUNCATED, rep);

      if (rep.is_zero())
        continue; // extra bits are only present for non-zero numbers.

      const auto sig = num.significand;
      num.significand = set_last_bits(sig, 1); // Smallest extra value.
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::EXACT, rep);
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::TRUNCATED, rep);
      num.significand = set_last_bits(sig, extra_bits); // Largest extra value.
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::EXACT, rep);
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::TRUNCATED, rep);
    }
  }
}

// Here we test materialization of a 'Number' back to an 'FPRep' with the
// 'AWAYZERO' rounding mode. This rounding mode will convert back to 'FPRep'
// only if there is no extra bit set and Truncation is 'EXACT', otherwise it
// will materialize as the next representable number.
TYPED_TEST(LlvmLibcFPBitsTest, NumberRoundAwayZero, FPTypes) {
  using StorageType = typename T::StorageType;
  using Number = typename T::Number;
  constexpr auto set_last_bits = [](StorageType value, int bits) {
    return value | ((StorageType(1) << bits) - StorageType(1));
  };
  const struct {
    FP initial;
    FP rounded;
  } TESTS[] = {
      {FP::ZERO, FP::MIN_SUBNORMAL},       //
      {FP::MAX_SUBNORMAL, FP::MIN_NORMAL}, //
      {FP::MAX_NORMAL, FP::INF},           //
  };
  for (Sign sign : all_signs) {
    for (auto tc : TESTS) {
      const T rep = make<T>(sign, tc.initial);
      const T rounded = make<T>(sign, tc.rounded);
      // Number with extra precision bits.
      Number num = rep.get_number().maximize_precision();
      const int extra_bits = Number::EXTRA_PRECISION + rep.is_subnormal();

      // Exact number converts back to rep.
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::EXACT, rep);
      // Non-exact numbers get rounded toward infinity.
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::TRUNCATED, rounded);

      if (rep.is_zero())
        continue; // extra bits are only present for non-zero numbers.

      const auto sig = num.significand;
      num.significand = set_last_bits(sig, 1); // Smallest extra value.
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::EXACT, rounded);
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::TRUNCATED, rounded);
      num.significand = set_last_bits(sig, extra_bits); // Largest extra value.
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::EXACT, rounded);
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::TRUNCATED, rounded);
    }
  }
}

// Here we test materialization of a 'Number' back to an 'FPRep' with the
// 'TONEAREST' rounding mode. This rounding mode will convert back to 'FPRep'
// only if there is no extra bit set and Truncation is 'EXACT', otherwise it
// will materialize as the next representable number.
TYPED_TEST(LlvmLibcFPBitsTest, NumberRoundToNearest, FPTypes) {
  using StorageType = typename T::StorageType;
  using Number = typename T::Number;
  constexpr auto set_last_bits = [](StorageType value, int bits) {
    return value | ((StorageType(1) << bits) - StorageType(1));
  };
  constexpr auto set_bit_at = [](StorageType value, int pos) {
    return value | (StorageType(1) << (pos - 1));
  };
  const struct {
    FP initial;
    FP rounded;
  } TESTS[] = {
      {FP::ZERO, FP::MIN_SUBNORMAL},       //
      {FP::MAX_SUBNORMAL, FP::MIN_NORMAL}, //
      {FP::MAX_NORMAL, FP::INF},           //
  };
  for (Sign sign : all_signs) {
    for (auto tc : TESTS) {
      const T rep = make<T>(sign, tc.initial);
      const T rounded = make<T>(sign, tc.rounded);
      Number num = rep.get_number().maximize_precision();
      const int extra_bits = Number::EXTRA_PRECISION + rep.is_subnormal();

      // Exact number converts back to rep.
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::EXACT, rep);
      // Non-exact numbers converts back to rep.
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::TRUNCATED, rep);

      if (rep.is_zero())
        continue; // extra bits are only present for non-zero numbers.

      const auto sig = num.significand;
      num.significand = set_last_bits(sig, 1); // Smallest extra value.
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::EXACT, rep);
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::TRUNCATED, rep);
      num.significand = set_last_bits(sig, extra_bits); // Largest extra value.
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::EXACT, rounded);
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::TRUNCATED, rounded);
      num.significand = set_bit_at(sig, extra_bits); // Half extra value.
      // We're exactly half-way between two numbers.
      // If exact we round toward zero.
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::EXACT, rep);
      // If truncated we round toward infinity.
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::TRUNCATED, rounded);
      // The next value will always round toward infinity.
      ++num.significand;
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::EXACT, rounded);
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::TRUNCATED, rounded);
    }
  }
}

TYPED_TEST(LlvmLibcFPBitsTest, SmallestNumber, FPTypes) {
  using StorageType = typename T::StorageType;
  using Number = typename T::Number;
  constexpr int32_t exponents[] = {INT32_MIN, INT32_MIN / 2};
  for (Sign sign : all_signs) {
    for (int32_t exponent : exponents) {
      Number num;
      num.sign = sign;
      num.exponent = exponent;
      num.significand = StorageType(1);

      const T zero = make<T>(sign, FP::ZERO);
      const T min = make<T>(sign, FP::MIN_SUBNORMAL);
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::EXACT, zero);
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::TRUNCATED, zero);
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::EXACT, min);
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::TRUNCATED, min);
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::EXACT, zero);
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::TRUNCATED, zero);
    }
  }
}

TYPED_TEST(LlvmLibcFPBitsTest, LargestNumber, FPTypes) {
  using StorageType = typename T::StorageType;
  using Number = typename T::Number;
  constexpr int32_t exponents[] = {INT32_MAX, INT32_MAX / 2};
  for (Sign sign : all_signs) {
    for (int32_t exponent : exponents) {
      Number num;
      num.sign = sign;
      num.exponent = exponent;
      num.significand = ~StorageType(0);

      const T max = make<T>(sign, FP::MAX_NORMAL);
      const T inf = make<T>(sign, FP::INF);
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::EXACT, max);
      ASSERT_MATERIALIZE_AS(num, Number::TOWARDZERO, Number::TRUNCATED, max);
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::EXACT, inf);
      ASSERT_MATERIALIZE_AS(num, Number::AWAYZERO, Number::TRUNCATED, inf);
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::EXACT, inf);
      ASSERT_MATERIALIZE_AS(num, Number::TONEAREST, Number::TRUNCATED, inf);
    }
  }
}

TEST(LlvmLibcFPBitsTest, FloatType) {
  using FloatBits = FPBits<float>;

  EXPECT_STREQ(LIBC_NAMESPACE::str(FloatBits::inf(Sign::POS)).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(FloatBits::inf(Sign::NEG)).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(FloatBits::signaling_nan()).c_str(),
               "(NaN)");

  FloatBits zero(0.0f);
  EXPECT_TRUE(zero.is_pos());
  EXPECT_EQ(zero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(zero.get_mantissa(), 0_u32);
  EXPECT_EQ(zero.uintval(), 0_u32);
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x00000000 = (S: 0, E: 0x0000, M: 0x00000000)");

  FloatBits negzero(-0.0f);
  EXPECT_TRUE(negzero.is_neg());
  EXPECT_EQ(negzero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(negzero.get_mantissa(), 0_u32);
  EXPECT_EQ(negzero.uintval(), 0x80000000_u32);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x80000000 = (S: 1, E: 0x0000, M: 0x00000000)");

  FloatBits one(1.0f);
  EXPECT_TRUE(one.is_pos());
  EXPECT_EQ(one.get_biased_exponent(), 0x7F_u16);
  EXPECT_EQ(one.get_mantissa(), 0_u32);
  EXPECT_EQ(one.uintval(), 0x3F800000_u32);
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3F800000 = (S: 0, E: 0x007F, M: 0x00000000)");

  FloatBits negone(-1.0f);
  EXPECT_TRUE(negone.is_neg());
  EXPECT_EQ(negone.get_biased_exponent(), 0x7F_u16);
  EXPECT_EQ(negone.get_mantissa(), 0_u32);
  EXPECT_EQ(negone.uintval(), 0xBF800000_u32);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBF800000 = (S: 1, E: 0x007F, M: 0x00000000)");

  FloatBits num(1.125f);
  EXPECT_TRUE(num.is_pos());
  EXPECT_EQ(num.get_biased_exponent(), 0x7F_u16);
  EXPECT_EQ(num.get_mantissa(), 0x00100000_u32);
  EXPECT_EQ(num.uintval(), 0x3F900000_u32);
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3F900000 = (S: 0, E: 0x007F, M: 0x00100000)");

  FloatBits negnum(-1.125f);
  EXPECT_TRUE(negnum.is_neg());
  EXPECT_EQ(negnum.get_biased_exponent(), 0x7F_u16);
  EXPECT_EQ(negnum.get_mantissa(), 0x00100000_u32);
  EXPECT_EQ(negnum.uintval(), 0xBF900000_u32);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBF900000 = (S: 1, E: 0x007F, M: 0x00100000)");

  FloatBits quiet_nan = FloatBits::quiet_nan();
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
}

TEST(LlvmLibcFPBitsTest, DoubleType) {
  using DoubleBits = FPBits<double>;

  EXPECT_STREQ(LIBC_NAMESPACE::str(DoubleBits::inf(Sign::POS)).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(DoubleBits::inf(Sign::NEG)).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(DoubleBits::signaling_nan()).c_str(),
               "(NaN)");

  DoubleBits zero(0.0);
  EXPECT_TRUE(zero.is_pos());
  EXPECT_EQ(zero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(zero.get_mantissa(), 0_u64);
  EXPECT_EQ(zero.uintval(), 0_u64);
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x0000000000000000 = (S: 0, E: 0x0000, M: 0x0000000000000000)");

  DoubleBits negzero(-0.0);
  EXPECT_TRUE(negzero.is_neg());
  EXPECT_EQ(negzero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(negzero.get_mantissa(), 0_u64);
  EXPECT_EQ(negzero.uintval(), 0x8000000000000000_u64);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x8000000000000000 = (S: 1, E: 0x0000, M: 0x0000000000000000)");

  DoubleBits one(1.0);
  EXPECT_TRUE(one.is_pos());
  EXPECT_EQ(one.get_biased_exponent(), 0x03FF_u16);
  EXPECT_EQ(one.get_mantissa(), 0_u64);
  EXPECT_EQ(one.uintval(), 0x3FF0000000000000_u64);
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3FF0000000000000 = (S: 0, E: 0x03FF, M: 0x0000000000000000)");

  DoubleBits negone(-1.0);
  EXPECT_TRUE(negone.is_neg());
  EXPECT_EQ(negone.get_biased_exponent(), 0x03FF_u16);
  EXPECT_EQ(negone.get_mantissa(), 0_u64);
  EXPECT_EQ(negone.uintval(), 0xBFF0000000000000_u64);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBFF0000000000000 = (S: 1, E: 0x03FF, M: 0x0000000000000000)");

  DoubleBits num(1.125);
  EXPECT_TRUE(num.is_pos());
  EXPECT_EQ(num.get_biased_exponent(), 0x03FF_u16);
  EXPECT_EQ(num.get_mantissa(), 0x0002000000000000_u64);
  EXPECT_EQ(num.uintval(), 0x3FF2000000000000_u64);
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3FF2000000000000 = (S: 0, E: 0x03FF, M: 0x0002000000000000)");

  DoubleBits negnum(-1.125);
  EXPECT_TRUE(negnum.is_neg());
  EXPECT_EQ(negnum.get_biased_exponent(), 0x03FF_u16);
  EXPECT_EQ(negnum.get_mantissa(), 0x0002000000000000_u64);
  EXPECT_EQ(negnum.uintval(), 0xBFF2000000000000_u64);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBFF2000000000000 = (S: 1, E: 0x03FF, M: 0x0002000000000000)");

  DoubleBits quiet_nan = DoubleBits::quiet_nan();
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
}

#ifdef LIBC_TARGET_ARCH_IS_X86
TEST(LlvmLibcFPBitsTest, X86LongDoubleType) {
  using LongDoubleBits = FPBits<long double>;

  if constexpr (sizeof(long double) == sizeof(double))
    return; // The tests for the "double" type cover for this case.

  EXPECT_STREQ(LIBC_NAMESPACE::str(LongDoubleBits::inf(Sign::POS)).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(LongDoubleBits::inf(Sign::NEG)).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(LongDoubleBits::signaling_nan()).c_str(),
               "(NaN)");

  LongDoubleBits zero(0.0l);
  EXPECT_TRUE(zero.is_pos());
  EXPECT_EQ(zero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(zero.get_mantissa(), 0_u128);
  EXPECT_EQ(zero.uintval(), 0_u128);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(zero).c_str(),
      "0x00000000000000000000000000000000 = "
      "(S: 0, E: 0x0000, I: 0, M: 0x00000000000000000000000000000000)");

  LongDoubleBits negzero(-0.0l);
  EXPECT_TRUE(negzero.is_neg());
  EXPECT_EQ(negzero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(negzero.get_mantissa(), 0_u128);
  EXPECT_EQ(negzero.uintval(), 0x8000'00000000'00000000_u128);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(negzero).c_str(),
      "0x00000000000080000000000000000000 = "
      "(S: 1, E: 0x0000, I: 0, M: 0x00000000000000000000000000000000)");

  LongDoubleBits one(1.0l);
  EXPECT_TRUE(one.is_pos());
  EXPECT_EQ(one.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(one.get_mantissa(), 0_u128);
  EXPECT_EQ(one.uintval(), 0x3FFF'80000000'00000000_u128);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(one).c_str(),
      "0x0000000000003FFF8000000000000000 = "
      "(S: 0, E: 0x3FFF, I: 1, M: 0x00000000000000000000000000000000)");

  LongDoubleBits negone(-1.0l);
  EXPECT_TRUE(negone.is_neg());
  EXPECT_EQ(negone.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(negone.get_mantissa(), 0_u128);
  EXPECT_EQ(negone.uintval(), 0xBFFF'80000000'00000000_u128);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(negone).c_str(),
      "0x000000000000BFFF8000000000000000 = "
      "(S: 1, E: 0x3FFF, I: 1, M: 0x00000000000000000000000000000000)");

  LongDoubleBits num(1.125l);
  EXPECT_TRUE(num.is_pos());
  EXPECT_EQ(num.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(num.get_mantissa(), 0x10000000'00000000_u128);
  EXPECT_EQ(num.uintval(), 0x3FFF'90000000'00000000_u128);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(num).c_str(),
      "0x0000000000003FFF9000000000000000 = "
      "(S: 0, E: 0x3FFF, I: 1, M: 0x00000000000000001000000000000000)");

  LongDoubleBits negnum(-1.125l);
  EXPECT_TRUE(negnum.is_neg());
  EXPECT_EQ(negnum.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(negnum.get_mantissa(), 0x10000000'00000000_u128);
  EXPECT_EQ(negnum.uintval(), 0xBFFF'90000000'00000000_u128);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(negnum).c_str(),
      "0x000000000000BFFF9000000000000000 = "
      "(S: 1, E: 0x3FFF, I: 1, M: 0x00000000000000001000000000000000)");

  LongDoubleBits quiet_nan = LongDoubleBits::quiet_nan();
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
}
#else
TEST(LlvmLibcFPBitsTest, LongDoubleType) {
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  return; // The tests for the "double" type cover for this case.
#else
  using LongDoubleBits = FPBits<long double>;

  EXPECT_STREQ(LIBC_NAMESPACE::str(LongDoubleBits::inf(Sign::POS)).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(LongDoubleBits::inf(Sign::NEG)).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(LongDoubleBits::signaling_nan()).c_str(),
               "(NaN)");

  LongDoubleBits zero(0.0l);
  EXPECT_TRUE(zero.is_pos());
  EXPECT_EQ(zero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(zero.get_mantissa(), 0_u128);
  EXPECT_EQ(zero.uintval(), 0_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x00000000000000000000000000000000 = "
               "(S: 0, E: 0x0000, M: 0x00000000000000000000000000000000)");

  LongDoubleBits negzero(-0.0l);
  EXPECT_TRUE(negzero.is_neg());
  EXPECT_EQ(negzero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(negzero.get_mantissa(), 0_u128);
  EXPECT_EQ(negzero.uintval(), 0x80000000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x80000000000000000000000000000000 = "
               "(S: 1, E: 0x0000, M: 0x00000000000000000000000000000000)");

  LongDoubleBits one(1.0l);
  EXPECT_TRUE(one.is_pos());
  EXPECT_EQ(one.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(one.get_mantissa(), 0_u128);
  EXPECT_EQ(one.uintval(), 0x3FFF0000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3FFF0000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  LongDoubleBits negone(-1.0l);
  EXPECT_TRUE(negone.is_neg());
  EXPECT_EQ(negone.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(negone.get_mantissa(), 0_u128);
  EXPECT_EQ(negone.uintval(), 0xBFFF0000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBFFF0000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  LongDoubleBits num(1.125l);
  EXPECT_TRUE(num.is_pos());
  EXPECT_EQ(num.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(num.get_mantissa(), 0x2000'00000000'00000000'00000000_u128);
  EXPECT_EQ(num.uintval(), 0x3FFF2000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3FFF2000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00002000000000000000000000000000)");

  LongDoubleBits negnum(-1.125l);
  EXPECT_TRUE(negnum.is_neg());
  EXPECT_EQ(negnum.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(negnum.get_mantissa(), 0x2000'00000000'00000000'00000000_u128);
  EXPECT_EQ(negnum.uintval(), 0xBFFF2000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBFFF2000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00002000000000000000000000000000)");

  LongDoubleBits quiet_nan = LongDoubleBits::quiet_nan();
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
#endif
}
#endif

#if defined(LIBC_TYPES_HAS_FLOAT128)
TEST(LlvmLibcFPBitsTest, Float128Type) {
  using Float128Bits = FPBits<float128>;

  EXPECT_STREQ(LIBC_NAMESPACE::str(Float128Bits::inf(Sign::POS)).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(Float128Bits::inf(Sign::NEG)).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(Float128Bits::signaling_nan()).c_str(),
               "(NaN)");

  Float128Bits zero = Float128Bits::zero(Sign::POS);
  EXPECT_TRUE(zero.is_pos());
  EXPECT_EQ(zero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(zero.get_mantissa(), 0_u128);
  EXPECT_EQ(zero.uintval(), 0_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x00000000000000000000000000000000 = "
               "(S: 0, E: 0x0000, M: 0x00000000000000000000000000000000)");

  Float128Bits negzero = Float128Bits::zero(Sign::NEG);
  EXPECT_TRUE(negzero.is_neg());
  EXPECT_EQ(negzero.get_biased_exponent(), 0_u16);
  EXPECT_EQ(negzero.get_mantissa(), 0_u128);
  EXPECT_EQ(negzero.uintval(), 0x80000000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x80000000000000000000000000000000 = "
               "(S: 1, E: 0x0000, M: 0x00000000000000000000000000000000)");

  Float128Bits one(float128(1.0));
  EXPECT_TRUE(one.is_pos());
  EXPECT_EQ(one.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(one.get_mantissa(), 0_u128);
  EXPECT_EQ(one.uintval(), 0x3FFF0000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3FFF0000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  Float128Bits negone(float128(-1.0));
  EXPECT_TRUE(negone.is_neg());
  EXPECT_EQ(negone.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(negone.get_mantissa(), 0_u128);
  EXPECT_EQ(negone.uintval(), 0xBFFF0000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBFFF0000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  Float128Bits num(float128(1.125));
  EXPECT_TRUE(num.is_pos());
  EXPECT_EQ(num.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(num.get_mantissa(), 0x2000'00000000'00000000'00000000_u128);
  EXPECT_EQ(num.uintval(), 0x3FFF2000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3FFF2000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00002000000000000000000000000000)");

  Float128Bits negnum(float128(-1.125));
  EXPECT_TRUE(negnum.is_neg());
  EXPECT_EQ(negnum.get_biased_exponent(), 0x3FFF_u16);
  EXPECT_EQ(negnum.get_mantissa(), 0x2000'00000000'00000000'00000000_u128);
  EXPECT_EQ(negnum.uintval(), 0xBFFF2000'00000000'00000000'00000000_u128);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBFFF2000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00002000000000000000000000000000)");

  Float128Bits quiet_nan = Float128Bits::quiet_nan();
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
}
#endif // LIBC_TYPES_HAS_FLOAT128
