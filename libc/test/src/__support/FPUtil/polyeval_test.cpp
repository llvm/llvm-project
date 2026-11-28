//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for polyeval, altpolyeval, loop_polyeval, loop_altpolyeval, and
/// estrin_polyeval.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/polyeval.h"
#include "src/__support/frac128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Frac128;
using LIBC_NAMESPACE::fputil::altpolyeval;
using LIBC_NAMESPACE::fputil::DoubleDouble;
using LIBC_NAMESPACE::fputil::DyadicFloat;
using LIBC_NAMESPACE::fputil::estrin_polyeval;
using LIBC_NAMESPACE::fputil::loop_altpolyeval;
using LIBC_NAMESPACE::fputil::loop_polyeval;
using LIBC_NAMESPACE::fputil::multiply_add;
using LIBC_NAMESPACE::fputil::polyeval;
using LIBC_NAMESPACE::fputil::sub_multiply;
using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

static constexpr RoundingMode ROUNDING_MODES[4] = {
    RoundingMode::Nearest,
    RoundingMode::Upward,
    RoundingMode::Downward,
    RoundingMode::TowardZero,
};

template <typename T>
class LlvmLibcPolyEvalTest : public LIBC_NAMESPACE::testing::Test {
public:
  void test_polyeval() {
    constexpr T C[6] = {
        T(0x1.921fb54442d18p0),  T(-0x1.5bf0a8b145769p1),
        T(0x1.62e42fefa39efp-1), T(-0x1.71547652b82fep-1),
        T(0x1.a01a01a01a01ap0),  T(-0x1.27e4fb7789f5cp0),
    };
    constexpr T XS[5] = {
        T(0.0),
        T(0.5),
        T(0x1.3579bdf13579bp-1),
        T(-0x1.2468ace02468bp-2),
        T(0x1.1111111111111p0),
    };

    for (RoundingMode mode : ROUNDING_MODES) {
      ForceRoundingMode r(mode);
      if (!r.success)
        continue;
      for (T raw_x : XS) {
        volatile T vx = raw_x;
        T x = vx;

        EXPECT_FP_EQ(C[0], polyeval(x, C[0]));
        EXPECT_FP_EQ(C[5], polyeval<5>(x, C));
        EXPECT_FP_EQ(C[5], loop_polyeval<5>(x, C));

        T h1 = multiply_add(x, C[1], C[0]);
        EXPECT_FP_EQ(h1, polyeval(x, C[0], C[1]));

        T h2_1 = multiply_add(x, C[2], C[1]);
        T h2 = multiply_add(x, h2_1, C[0]);
        EXPECT_FP_EQ(h2, polyeval(x, C[0], C[1], C[2]));

        T t5 = C[5];
        T t4 = multiply_add(x, t5, C[4]);
        T t3 = multiply_add(x, t4, C[3]);
        T t2 = multiply_add(x, t3, C[2]);
        T t1 = multiply_add(x, t2, C[1]);
        T t0 = multiply_add(x, t1, C[0]);

        EXPECT_FP_EQ(t0, polyeval(x, C[0], C[1], C[2], C[3], C[4], C[5]));
        EXPECT_FP_EQ(t0, polyeval(x, C));
        EXPECT_FP_EQ(t1, polyeval<1>(x, C));
        EXPECT_FP_EQ(t2, polyeval<2>(x, C));
        EXPECT_FP_EQ(t4, polyeval<4>(x, C));
        EXPECT_FP_EQ(t0, loop_polyeval(x, C));
        EXPECT_FP_EQ(t1, loop_polyeval<1>(x, C));
        EXPECT_FP_EQ(t2, loop_polyeval<2>(x, C));
        EXPECT_FP_EQ(t4, loop_polyeval<4>(x, C));
      }
    }
  }

  void test_altpolyeval() {
    constexpr T EXACT_C[5] = {T(2.0), T(1.5), T(0.75), T(0.25), T(1.125)};

    // Exact dyadic check at x = 0.5:
    // 2.0 - 0.5 * 1.5 + 0.25 * 0.75 - 0.125 * 0.25 + 0.0625 * 1.125
    // = (512 - 192 + 48 - 8 + 18) / 256 = 378 / 256 = 189 / 128 = 1.4765625.
    EXPECT_FP_EQ(T(1.4765625), altpolyeval(T(0.5), EXACT_C));
    EXPECT_FP_EQ(T(1.4765625), loop_altpolyeval(T(0.5), EXACT_C));
    EXPECT_FP_EQ(T(1.4765625), altpolyeval(T(0.5), EXACT_C[0], EXACT_C[1],
                                           EXACT_C[2], EXACT_C[3], EXACT_C[4]));

    constexpr T C[5] = {
        T(0x1.921fb54442d18p0),  T(0x1.5bf0a8b145769p0),
        T(0x1.62e42fefa39efp-1), T(0x1.71547652b82fep-2),
        T(0x1.a01a01a01a01ap-1),
    };
    constexpr T XS[5] = {
        T(0.0),
        T(0.5),
        T(0x1.3579bdf13579bp-1),
        T(-0x1.2468ace02468bp-2),
        T(0x1.7f3d5b97f3d5cp-1),
    };

    // Check constexpr coefficient array in round-to-nearest.
    {
      ForceRoundingMode r(RoundingMode::Nearest);
      if (r.success) {
        volatile T vx = XS[2];
        T x = vx;
        T t4 = C[4];
        T t3 = sub_multiply(C[3], x, t4);
        T t2 = sub_multiply(C[2], x, t3);
        T t1 = sub_multiply(C[1], x, t2);
        T t0 = sub_multiply(C[0], x, t1);
        EXPECT_FP_EQ(t0, altpolyeval(x, C));
        EXPECT_FP_EQ(t0, loop_altpolyeval(x, C));
      }
    }

    // Load coefficients at runtime when testing directed rounding modes so
    // compilers without -frounding-math do not fold -C[4] differently between
    // unrolled altpolyeval and non-unrolled loop_altpolyeval.
    volatile T vc[5] = {C[0], C[1], C[2], C[3], C[4]};
    const T rc[5] = {vc[0], vc[1], vc[2], vc[3], vc[4]};

    for (RoundingMode mode : ROUNDING_MODES) {
      ForceRoundingMode r(mode);
      if (!r.success)
        continue;
      for (T raw_x : XS) {
        volatile T vx = raw_x;
        T x = vx;

        EXPECT_FP_EQ(rc[0], altpolyeval(x, rc[0]));
        EXPECT_FP_EQ(rc[4], altpolyeval<4>(x, rc));
        EXPECT_FP_EQ(rc[4], loop_altpolyeval<4>(x, rc));

        T t4 = rc[4];
        T t3 = sub_multiply(rc[3], x, t4);
        T t2 = sub_multiply(rc[2], x, t3);
        T t1 = sub_multiply(rc[1], x, t2);
        T t0 = sub_multiply(rc[0], x, t1);

        EXPECT_FP_EQ(t0, altpolyeval(x, rc[0], rc[1], rc[2], rc[3], rc[4]));
        EXPECT_FP_EQ(t0, altpolyeval(x, rc));
        EXPECT_FP_EQ(t1, altpolyeval<1>(x, rc));
        EXPECT_FP_EQ(t2, altpolyeval<2>(x, rc));
        EXPECT_FP_EQ(t3, altpolyeval<3>(x, rc));
        EXPECT_FP_EQ(t0, loop_altpolyeval(x, rc));
        EXPECT_FP_EQ(t1, loop_altpolyeval<1>(x, rc));
        EXPECT_FP_EQ(t2, loop_altpolyeval<2>(x, rc));
        EXPECT_FP_EQ(t3, loop_altpolyeval<3>(x, rc));
      }
    }
  }

  void test_estrin_polyeval() {
    constexpr T C[13] = {
        T(0x1.921fb54442d18p0),  T(-0x1.5bf0a8b145769p-1),
        T(0x1.62e42fefa39efp-1), T(-0x1.71547652b82fep-2),
        T(0x1.a01a01a01a01ap-2), T(-0x1.27e4fb7789f5cp-2),
        T(0x1.1eed8eff8d898p-1), T(-0x1.b803473f7ad0fp-2),
        T(0x1.3b2ab6fba4e77p-1), T(-0x1.d8e64b8d4ddadp-2),
        T(0x1.0c152382d7365p-1), T(-0x1.8f8b83c69a60bp-2),
        T(0x1.7c5db1052981bp-2),
    };
    constexpr T XS[5] = {
        T(0.0),
        T(0.5),
        T(0x1.3579bdf13579bp-1),
        T(-0x1.2468ace02468bp-2),
        T(0x1.7f3d5b97f3d5cp-1),
    };

    for (RoundingMode mode : ROUNDING_MODES) {
      ForceRoundingMode r(mode);
      if (!r.success)
        continue;
      for (T raw_x : XS) {
        volatile T vx = raw_x;
        T x = vx;

        T x2 = x * x;
        T x4 = x2 * x2;
        T x8 = x4 * x4;

        // Length 1 (degree 0).
        EXPECT_FP_EQ(C[0], estrin_polyeval(x, C[0]));
        EXPECT_FP_EQ(C[12], estrin_polyeval<12>(x, C));

        // Length 2 (degree 1): a0 + x * a1.
        T p01 = multiply_add(x, C[1], C[0]);
        EXPECT_FP_EQ(p01, estrin_polyeval(x, C[0], C[1]));

        // Length 3 (degree 2, 1 + 2): a0 + x * (a1 + x * a2).
        T p12 = multiply_add(x, C[2], C[1]);
        T e3 = multiply_add(x, p12, C[0]);
        EXPECT_FP_EQ(e3, estrin_polyeval(x, C[0], C[1], C[2]));

        // Length 4 (degree 3, 2 + 2): (a0 + x * a1) + x2 * (a2 + x * a3).
        T p23 = multiply_add(x, C[3], C[2]);
        T e4 = multiply_add(x2, p23, p01);
        EXPECT_FP_EQ(e4, estrin_polyeval(x, C[0], C[1], C[2], C[3]));

        // Length 5 (degree 4, 1 + 4): a0 + x * ((a1 + x*a2) + x2 * (a3 +
        // x*a4)).
        T p34 = multiply_add(x, C[4], C[3]);
        T e1_4 = multiply_add(x2, p34, p12);
        T e5 = multiply_add(x, e1_4, C[0]);
        EXPECT_FP_EQ(e5, estrin_polyeval(x, C[0], C[1], C[2], C[3], C[4]));

        // Length 6 (degree 5, 2 + 4): (a0 + x*a1) + x2 * E4(a2..a5).
        T p45 = multiply_add(x, C[5], C[4]);
        T e2_5 = multiply_add(x2, p45, p23);
        T e6 = multiply_add(x2, e2_5, p01);
        EXPECT_FP_EQ(e6,
                     estrin_polyeval(x, C[0], C[1], C[2], C[3], C[4], C[5]));

        // Length 7 (degree 6, 4 + 3): E4(a0..a3) + x4 * (a4 + x*(a5 + x*a6)).
        T p56 = multiply_add(x, C[6], C[5]);
        T e4_6 = multiply_add(x, p56, C[4]);
        T e7 = multiply_add(x4, e4_6, e4);
        EXPECT_FP_EQ(
            e7, estrin_polyeval(x, C[0], C[1], C[2], C[3], C[4], C[5], C[6]));

        // Length 8 (degree 7, 4 + 4): E4(a0..a3) + x4 * E4(a4..a7).
        T p67 = multiply_add(x, C[7], C[6]);
        T e4_7 = multiply_add(x2, p67, p45);
        T e8 = multiply_add(x4, e4_7, e4);
        EXPECT_FP_EQ(e8, estrin_polyeval(x, C[0], C[1], C[2], C[3], C[4], C[5],
                                         C[6], C[7]));

        // Length 9 (degree 8, 1 + 8): a0 + x * E8(a1..a8).
        T p78 = multiply_add(x, C[8], C[7]);
        T e5_8 = multiply_add(x2, p78, p56);
        T e1_8 = multiply_add(x4, e5_8, e1_4);
        T e9 = multiply_add(x, e1_8, C[0]);
        EXPECT_FP_EQ(e9, estrin_polyeval(x, C[0], C[1], C[2], C[3], C[4], C[5],
                                         C[6], C[7], C[8]));

        // Slice with OFFSET = 7 (6 elements: C[7..12], 2 + 4).
        T p9_10 = multiply_add(x, C[10], C[9]);
        T p11_12 = multiply_add(x, C[12], C[11]);
        T e9_12 = multiply_add(x2, p11_12, p9_10);
        T s7_12 = multiply_add(x2, e9_12, p78);
        EXPECT_FP_EQ(s7_12, estrin_polyeval<7>(x, C));

        // Full array C (length 13, 8 + 5): E8(a0..a7) + x8 * (a8 +
        // x*E4(a9..a12)).
        T e8_12 = multiply_add(x, e9_12, C[8]);
        T e13 = multiply_add(x8, e8_12, e8);
        EXPECT_FP_EQ(e13, estrin_polyeval(x, C));
      }
    }
  }

  // Verify with half-ULP inputs that wrong tree parenthesizations fail.
  void test_order_sensitivity() {
    ForceRoundingMode r(RoundingMode::Nearest);
    if (!r.success)
      return;

    // U = 2^(-p) = 0.5 * ulp(1.0), so RN(1.0 + U) = 1.0 (tie to even) whereas
    // RN(1.0 + c * U) = 1.0 + 2 * U for 1 < c < 3.
    constexpr T U = (sizeof(T) == sizeof(float)) ? T(0x1p-24) : T(0x1p-53);
    constexpr T ONE_PLUS_ULP = T(1.0) + T(2.0) * U;

    volatile T vx = T(0.5);
    T x = vx;
    T x2 = x * x;
    T x4 = x2 * x2;

    // LEN = 3: exact = 1 + 1.5 * U.
    //   Right-heavy a0 + x*(a1 + x*a2) = RN(1 + 1.5*U) = 1 + 2*U,
    //   whereas left-heavy (a0 + x*a1) + x2*a2 = RN(RN(1 + U) + 0.5*U) = 1.
    constexpr T C3[3] = {T(1.0), T(2.0) * U, T(2.0) * U};
    T left3_lo = multiply_add(x, C3[1], C3[0]);
    T left3 = multiply_add(x2, C3[2], left3_lo);
    EXPECT_FP_EQ(T(1.0), left3);
    EXPECT_FP_EQ(ONE_PLUS_ULP, polyeval(x, C3));
    EXPECT_FP_EQ(ONE_PLUS_ULP, loop_polyeval(x, C3));
    EXPECT_FP_EQ(ONE_PLUS_ULP, estrin_polyeval(x, C3));

    // LEN = 4: exact = 1 + 1.25 * U.
    //   Horner: a2 + x*a3 = U -> a1 + x*U = 2.5*U -> RN(1 + 1.25*U) = 1 + 2*U.
    //   Estrin (2 + 2): RN(1 + U) = 1, p23 = U -> RN(1 + 0.25*U) = 1.
    constexpr T C4[4] = {T(1.0), T(2.0) * U, T(2.0) * U, T(-2.0) * U};
    EXPECT_FP_EQ(ONE_PLUS_ULP, polyeval(x, C4));
    EXPECT_FP_EQ(ONE_PLUS_ULP, loop_polyeval(x, C4));
    EXPECT_FP_EQ(T(1.0), estrin_polyeval(x, C4));

    // LEN = 5: exact = 1 + 1.5 * U.
    //   Right-heavy (1 + 4): E4(2*U, 0, 0, 8*U) = 3*U -> RN(1 + 1.5*U) = 1+2*U.
    //   Left-heavy (4 + 1): E4(1, 2*U, 0, 0) = 1 -> RN(1 + 0.5*U) = 1.
    constexpr T C5[5] = {T(1.0), T(2.0) * U, T(0.0), T(0.0), T(8.0) * U};
    T left5_lo = estrin_polyeval(x, C5[0], C5[1], C5[2], C5[3]);
    T left5 = multiply_add(x4, C5[4], left5_lo);
    EXPECT_FP_EQ(T(1.0), left5);
    EXPECT_FP_EQ(ONE_PLUS_ULP, estrin_polyeval(x, C5));

    // LEN = 6: exact = 1 + 1.5 * U.
    //   Right-heavy (2 + 4): p01 = 1, E4(4*U, 0, 8*U, 0) = 6*U
    //                        -> RN(1 + 0.25 * 6*U) = 1 + 2*U.
    //   Left-heavy (4 + 2): E4(1, 0, 4*U, 0) = RN(1 + U) = 1
    //                       -> RN(1 + 0.0625 * 8*U) = 1.
    constexpr T C6[6] = {T(1.0), T(0.0),     T(4.0) * U,
                         T(0.0), T(8.0) * U, T(0.0)};
    T left6_lo = estrin_polyeval(x, C6[0], C6[1], C6[2], C6[3]);
    T left6_hi = multiply_add(x, C6[5], C6[4]);
    T left6 = multiply_add(x4, left6_hi, left6_lo);
    EXPECT_FP_EQ(T(1.0), left6);
    EXPECT_FP_EQ(ONE_PLUS_ULP, estrin_polyeval(x, C6));

    // LEN = 7: 4 + 3 with right-heavy hi = a4 + x*(a5 + x*a6) = 1 + 2*U,
    //   whereas left-heavy hi = (a4 + x*a5) + x2*a6 = 1.
    constexpr T C7[7] = {T(0.0), T(0.0),     T(0.0),    T(0.0),
                         T(1.0), T(2.0) * U, T(2.0) * U};
    EXPECT_FP_EQ(T(0.0625) * ONE_PLUS_ULP, estrin_polyeval(x, C7));
  }

  void test_constexpr() {
#if LIBC_HAS_BUILTIN_IS_CONSTANT_EVALUATED
    constexpr T C[4] = {T(2.0), T(1.5), T(0.75), T(0.25)};
    // At x = 0.5: 2.0 + 0.5*1.5 + 0.25*0.75 + 0.125*0.25 = 2.96875;
    //             2.0 - 0.5*1.5 + 0.25*0.75 - 0.125*0.25 = 1.40625.
    static_assert(polyeval(T(0.5), C[0], C[1], C[2], C[3]) == T(2.96875));
    static_assert(polyeval(T(0.5), C) == T(2.96875));
    static_assert(loop_polyeval(T(0.5), C) == T(2.96875));
    static_assert(estrin_polyeval(T(0.5), C) == T(2.96875));
    static_assert(altpolyeval(T(0.5), C) == T(1.40625));
    static_assert(loop_altpolyeval(T(0.5), C) == T(1.40625));
#endif
  }
};

using PolyEvalFPTypes = LIBC_NAMESPACE::testing::Types<float, double>;

TYPED_TEST_SUITE(LlvmLibcPolyEvalTest, PolyEvalFPTypes);

TYPED_TEST(LlvmLibcPolyEvalTest, PolyEval) { this->test_polyeval(); }

TYPED_TEST(LlvmLibcPolyEvalTest, AltPolyEval) { this->test_altpolyeval(); }

TYPED_TEST(LlvmLibcPolyEvalTest, EstrinPolyEval) {
  this->test_estrin_polyeval();
}

TYPED_TEST(LlvmLibcPolyEvalTest, OrderSensitivity) {
  this->test_order_sensitivity();
}

TYPED_TEST(LlvmLibcPolyEvalTest, ConstexprEval) { this->test_constexpr(); }

TEST(LlvmLibcPolyEvalStructTest, DoubleDoubleHorner) {
  const DoubleDouble x{0x1.3p-56, 0x1.3579bdf13579bp-2};
  const DoubleDouble c[4] = {
      {0x1.1p-54, 0x1.921fb54442d18p0},
      {-0x1.5p-55, 0x1.5bf0a8b145769p-1},
      {0x1.7p-56, 0x1.62e42fefa39efp-2},
      {-0x1.9p-57, 0x1.71547652b82fep-3},
  };

  DoubleDouble t2 = multiply_add(x, c[3], c[2]);
  DoubleDouble t1 = multiply_add(x, t2, c[1]);
  DoubleDouble t0 = multiply_add(x, t1, c[0]);

  DoubleDouble r1 = polyeval(x, c[0], c[1], c[2], c[3]);
  DoubleDouble r2 = polyeval(x, c);
  DoubleDouble r3 = loop_polyeval(x, c);
  DoubleDouble r_slice = polyeval<1>(x, c);
  DoubleDouble r_loop_slice = loop_polyeval<1>(x, c);

  EXPECT_FP_EQ(t0.hi, r1.hi);
  EXPECT_FP_EQ(t0.lo, r1.lo);
  EXPECT_FP_EQ(t0.hi, r2.hi);
  EXPECT_FP_EQ(t0.lo, r2.lo);
  EXPECT_FP_EQ(t0.hi, r3.hi);
  EXPECT_FP_EQ(t0.lo, r3.lo);
  EXPECT_FP_EQ(t1.hi, r_slice.hi);
  EXPECT_FP_EQ(t1.lo, r_slice.lo);
  EXPECT_FP_EQ(t1.hi, r_loop_slice.hi);
  EXPECT_FP_EQ(t1.lo, r_loop_slice.lo);

  DoubleDouble s2 = sub_multiply(c[2], x, c[3]);
  DoubleDouble s1 = sub_multiply(c[1], x, s2);
  DoubleDouble s0 = sub_multiply(c[0], x, s1);

  DoubleDouble a1 = altpolyeval(x, c[0], c[1], c[2], c[3]);
  DoubleDouble a2 = altpolyeval(x, c);
  DoubleDouble a3 = loop_altpolyeval(x, c);
  EXPECT_FP_EQ(s0.hi, a1.hi);
  EXPECT_FP_EQ(s0.lo, a1.lo);
  EXPECT_FP_EQ(s0.hi, a2.hi);
  EXPECT_FP_EQ(s0.lo, a2.lo);
  EXPECT_FP_EQ(s0.hi, a3.hi);
  EXPECT_FP_EQ(s0.lo, a3.lo);
}

TEST(LlvmLibcPolyEvalStructTest, DyadicFloatEval) {
  using DFloat128 = DyadicFloat<128>;
  using Mant = DFloat128::MantissaType;
  constexpr DFloat128 X(
      Sign::POS, -129,
      Mant({0x1357'9bdf'2468'ace1ULL, 0x9abc'def0'1234'5678ULL}));
  constexpr DFloat128 C[4] = {
      DFloat128(Sign::POS, -127,
                Mant({0xfedc'ba98'7654'3211ULL, 0xc90f'daa2'2168'c234ULL})),
      DFloat128(Sign::POS, -128,
                Mant({0x1122'3344'5566'7789ULL, 0xadfc'8458'a2bb'4a9aULL})),
      DFloat128(Sign::POS, -129,
                Mant({0x99aa'bbcc'ddee'ff01ULL, 0xb172'17f7'd1cf'79abULL})),
      DFloat128(Sign::POS, -130,
                Mant({0x5555'5555'5555'5557ULL, 0xb8aa'3b29'5c17'f0bbULL})),
  };

  constexpr DFloat128 T2 = multiply_add(X, C[3], C[2]);
  constexpr DFloat128 T1 = multiply_add(X, T2, C[1]);
  constexpr DFloat128 T0 = multiply_add(X, T1, C[0]);

  constexpr DFloat128 P1 = polyeval(X, C[0], C[1], C[2], C[3]);
  constexpr DFloat128 P2 = polyeval(X, C);
  constexpr DFloat128 P3 = loop_polyeval(X, C);
  static_assert(T0.mantissa == P1.mantissa && T0.exponent == P1.exponent);
  static_assert(T0.mantissa == P2.mantissa && T0.exponent == P2.exponent);
  static_assert(T0.mantissa == P3.mantissa && T0.exponent == P3.exponent);

  constexpr DFloat128 S2 = sub_multiply(C[2], X, C[3]);
  constexpr DFloat128 S1 = sub_multiply(C[1], X, S2);
  constexpr DFloat128 S0 = sub_multiply(C[0], X, S1);

  constexpr DFloat128 A1 = altpolyeval(X, C[0], C[1], C[2], C[3]);
  constexpr DFloat128 A2 = altpolyeval(X, C);
  constexpr DFloat128 A3 = loop_altpolyeval(X, C);
  static_assert(S0.mantissa == A1.mantissa && S0.exponent == A1.exponent);
  static_assert(S0.mantissa == A2.mantissa && S0.exponent == A2.exponent);
  static_assert(S0.mantissa == A3.mantissa && S0.exponent == A3.exponent);

  EXPECT_FP_EQ(static_cast<double>(T0), static_cast<double>(P2));
  EXPECT_FP_EQ(static_cast<double>(S0), static_cast<double>(A2));
}

TEST(LlvmLibcPolyEvalStructTest, Frac128Eval) {
  // Exact dyadic check in [0, 1):
  // 0.5 + 0.25*0.25 + 0.0625*0.125 + 0.015625*0.0625 = 585 / 1024.
  constexpr Frac128 X_EXACT({0, 0x4000'0000'0000'0000ULL}); // 0.25
  constexpr Frac128 C_EXACT[4] = {
      Frac128({0, 0x8000'0000'0000'0000ULL}), // 0.5
      Frac128({0, 0x4000'0000'0000'0000ULL}), // 0.25
      Frac128({0, 0x2000'0000'0000'0000ULL}), // 0.125
      Frac128({0, 0x1000'0000'0000'0000ULL}), // 0.0625
  };
  constexpr Frac128 EXPECTED_SUM({0, (585ULL << 54)});
  static_assert(EXPECTED_SUM == polyeval(X_EXACT, C_EXACT));
  static_assert(EXPECTED_SUM == estrin_polyeval(X_EXACT, C_EXACT));

  // Full-128-bit values where truncation differs between Horner and Estrin.
  constexpr Frac128 X({0xffff'ffff'ffff'ffffULL, 0x5555'5555'5555'5555ULL});
  constexpr Frac128 C[4] = {
      Frac128({0xfedc'ba98'7654'3211ULL, 0x6a09'e667'f3bc'c908ULL}),
      Frac128({0xffff'ffff'ffff'ffffULL, 0x3b67'ae85'84ca'a73bULL}),
      Frac128({0xffff'ffff'ffff'ffffULL, 0x2c6e'f372'fe94'f82bULL}),
      Frac128({0xdead'beef'cafe'bab3ULL, 0x154f'f53a'5f1d'36f1ULL}),
  };

  constexpr Frac128 H2 = multiply_add(X, C[3], C[2]);
  constexpr Frac128 H1 = multiply_add(X, H2, C[1]);
  constexpr Frac128 H0 = multiply_add(X, H1, C[0]);

  constexpr Frac128 X2 = X * X;
  constexpr Frac128 E_LO = multiply_add(X, C[1], C[0]);
  constexpr Frac128 E_HI = multiply_add(X, C[3], C[2]);
  constexpr Frac128 E0 = multiply_add(X2, E_HI, E_LO);

  // Confirm Horner and Estrin produce distinct 128-bit results on C.
  static_assert(!(H0 == E0));

  static_assert(H0 == polyeval(X, C[0], C[1], C[2], C[3]));
  static_assert(H0 == polyeval(X, C));
  static_assert(H0 == loop_polyeval(X, C));
  static_assert(H1 == polyeval<1>(X, C));
  static_assert(H1 == loop_polyeval<1>(X, C));
  static_assert(H1 == estrin_polyeval<1>(X, C));
  static_assert(E0 == estrin_polyeval(X, C[0], C[1], C[2], C[3]));
  static_assert(E0 == estrin_polyeval(X, C));

  EXPECT_TRUE(H0 == polyeval(X, C));
  EXPECT_TRUE(H0 == loop_polyeval(X, C));
  EXPECT_TRUE(E0 == estrin_polyeval(X, C));
  EXPECT_TRUE(H1 == estrin_polyeval<1>(X, C));

  constexpr Frac128 A2 = sub_multiply(C[2], X, C[3]);
  constexpr Frac128 A1 = sub_multiply(C[1], X, A2);
  constexpr Frac128 A0 = sub_multiply(C[0], X, A1);
  static_assert(A0 == altpolyeval(X, C[0], C[1], C[2], C[3]));
  static_assert(A0 == altpolyeval(X, C));
  static_assert(A0 == loop_altpolyeval(X, C));
  EXPECT_TRUE(A0 == altpolyeval(X, C));
  EXPECT_TRUE(A0 == loop_altpolyeval(X, C));
}
