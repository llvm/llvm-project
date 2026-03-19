//===----------------------------------------------------------------------===//
// Float128 Tests
//
// These tests validate the basic integration of the float128 type with:
//   - type traits (is_floating_point)
//   - FPBits functionality
//   - Float128 casting
// The goal is to ensure that both the type alias (float128) and the fallback
// implementation behave consistently with other floating-point types.
//===----------------------------------------------------------------------===//
#include "src/__support/FPUtil/cast.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/Test.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/float128.h"

using LIBC_NAMESPACE::fputil::cast;
using LIBC_NAMESPACE::fputil::Float128;

// Test: float128 is recognized as a floating-point type.
TEST(LlvmLibcTypeTraitsTest, Float128IsFloatingPoint) {
  using LIBC_NAMESPACE::cpp::is_floating_point_v;

  EXPECT_TRUE(is_floating_point_v<float128>);
}

// Test: Basic FPBits usage with float128 default initialization.
// Verifies zero initialization and basic classification APIs.
TEST(LlvmLibcFPBitsTest, Float128BasicUsage) {
  using LIBC_NAMESPACE::fputil::FPBits;

  float128 x{}; // Default-initialized to zero
  FPBits<float128> bits(x);

  EXPECT_TRUE(bits.is_zero());
  EXPECT_TRUE(bits.is_finite());
  EXPECT_FALSE(bits.is_nan());
  EXPECT_FALSE(bits.is_inf());
}

// Test: Construct FPBits<float128> directly from raw bits.
// Verifies that zero bit pattern is interpreted correctly.
TEST(LlvmLibcFPBitsTest, Float128FromBits) {
  using LIBC_NAMESPACE::fputil::FPBits;
  UInt128 raw = 0;
  FPBits<float128> bits(raw);

  EXPECT_TRUE(bits.is_zero());
}

// Test: Special values (infinity and NaN) for float128.
// Ensures FPBits builders work correctly for float128.
TEST(LlvmLibcFPBitsTest, Float128SpecialValues) {
  using LIBC_NAMESPACE::fputil::FPBits;

  auto inf = FPBits<float128>::inf();
  EXPECT_TRUE(inf.is_inf());

  auto nan = FPBits<float128>::quiet_nan();
  EXPECT_TRUE(nan.is_nan());
}

//Test float to float128 casting
TEST(LlvmLibcCastTest, FloatToFloat128ToFloat) {
  float x = 1.25f;
  float128 q = cast<float128>(x);
  float y = cast<float>(q);
  EXPECT_TRUE(x == y);
}

//Test double -> float128 -> double casting
TEST(LlvmLibcCastTest, DoubleToFloat128ToDouble) {
  double x = 1.5;
  float128 q = cast<float128>(x);
  double y = cast<double>(q);
  EXPECT_TRUE(x == y);
}

//Test bfloat16 -> float128 casting
TEST(LlvmLibcCastTest, bfloat16ToFloat128Tobfloat16) {
  bfloat16 x = cast<bfloat16>(0.1);
  float128 q = cast<float128>(x);
  bfloat16 y = cast<bfloat16>(q);
  EXPECT_TRUE(x == y);
}

TEST(LlvmLibcCastTest, RoundingBehavior) {
  double x = 0.1;
  float128 q = cast<float128>(x);
  double y = cast<double>(q);
  EXPECT_TRUE(x == y);
}

TEST(LlvmLibcCastTest, ZeroAndNegativeZero) {
  using LIBC_NAMESPACE::fputil::cast;
  double pos_zero = 0.0;
  double neg_zero = -0.0;

  float128 q1 = cast<float128>(pos_zero);
  float128 q2 = cast<float128>(neg_zero);

  double y1 = cast<double>(q1);
  double y2 = cast<double>(q2);

  EXPECT_TRUE(y1 == 0.0);
  EXPECT_TRUE(y2 == 0.0);
  EXPECT_TRUE(__builtin_signbit(y2) != 0);
}

TEST(LlvmLibcCastTest, SpecialValues) {
  using LIBC_NAMESPACE::fputil::cast;

  double inf = __builtin_inf();
  double nan = __builtin_nan("");

  float128 q_inf = cast<float128>(inf);
  float128 q_nan = cast<float128>(nan);

  double y_inf = cast<double>(q_inf);
  double y_nan = cast<double>(q_nan);

  EXPECT_TRUE(__builtin_isinf(y_inf) != 0);
  EXPECT_TRUE(__builtin_isnan(y_nan) != 0);
}

//test operators
TEST(LlvmLibcFloat128Test, BasicArithmetic) {
  float128 a = cast<float128>(1.5);
  float128 b = cast<float128>(2.0);

  Float128 x(a);
  Float128 y(b);

  EXPECT_TRUE((x + y) == cast<float128>(3.5));
  EXPECT_TRUE((x - y) == cast<float128>(-0.5));
  EXPECT_TRUE((x * y) == cast<float128>(3.0));
  EXPECT_TRUE((x / y) == cast<float128>(0.75));
}

TEST(LlvmLibcFloat128Test, ZeroBehavior) {
  float128 pos_zero = cast<float128>(0.0);
  float128 neg_zero = cast<float128>(-0.0);

  Float128 x(pos_zero);
  Float128 y(neg_zero);

  float128 r1 = x + y;
  float128 r2 = x - y;

  EXPECT_TRUE(r1 == 0.0);
  EXPECT_TRUE(r2 == 0.0);
  EXPECT_TRUE(__builtin_signbit(r2) == 0);
}

TEST(LlvmLibcFloat128Test, SpecialValues) {
  float128 inf = cast<float128>(__builtin_inf());
  float128 nan = cast<float128>(__builtin_nan(""));

  Float128 x(inf);
  Float128 y(nan);

  EXPECT_TRUE(__builtin_isinf(x + Float128(cast<float128>(1.0))) != 0);
  EXPECT_TRUE(__builtin_isnan(y + Float128(cast<float128>(1.0))) != 0);
}

TEST(LlvmLibcFloat128Test, PrecisionSanity) {
  float128 a = cast<float128>(0.1);
  float128 b = cast<float128>(0.2);

  Float128 x(a);
  Float128 y(b);

  float128 r = x + y;

  EXPECT_TRUE(r == a + b);
}

TEST(LlvmLibcFloat128Test, RoundTripConsistency) {
  float128 a = cast<float128>(1.25);

  Float128 x(a);
  float128 r = x + Float128(cast<float128>(0.0));

  EXPECT_TRUE(r == a);
}