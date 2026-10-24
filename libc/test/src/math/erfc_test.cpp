#include "src/math/erfc.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcErfcTest = LIBC_NAMESPACE::testing::FPTest;

TEST_F(LlvmLibcErfcTest, SpecialNumbers) {
  DECLARE_SPECIAL_CONSTANTS(double);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::erfc(aNaN));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::erfc(inf));
  EXPECT_FP_EQ(2.0, LIBC_NAMESPACE::erfc(neg_inf));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::erfc(zero));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::erfc(neg_zero));
}
