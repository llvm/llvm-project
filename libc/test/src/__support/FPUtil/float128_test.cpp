//===----------------------------------------------------------------------===//
// Float128 Tests
//
// These tests validate the basic integration of the float128 type with:
//   - type traits (is_floating_point)
//   - FPBits functionality
//
// The goal is to ensure that both the type alias (float128) and the fallback
// implementation behave consistently with other floating-point types.
//===----------------------------------------------------------------------===//
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/fpbits_str.h"
#include "src/__support/big_int.h"
#include "src/__support/integer_literals.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/sign.h" // Sign
#include "test/UnitTest/Test.h"
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