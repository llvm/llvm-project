//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "test/UnitTest/Test.h"

namespace {

using Vector = LIBC_NAMESPACE::cpp::simd<float>;
constexpr size_t LANES =
    LIBC_NAMESPACE::cpp::internal::native_vector_size<float>;
float identity(float x) { return x; }
LIBC_VECTOR_PCS Vector vector_identity(Vector x) { return x; }
using ExactChecker = UnaryOpChecker<float, float, identity, vector_identity>;
using TolerantChecker =
    UnaryOpChecker<float, float, identity, vector_identity, 1>;
using LlvmLibcMathvecCheckerTest = LIBC_NAMESPACE::testing::FPTest<float>;

} // namespace

TEST_F(LlvmLibcMathvecCheckerTest, MatchingLanes) {
  EXPECT_EQ(ExactChecker::first_mismatching_lane(1.0f, Vector(1.0f)), LANES);
  EXPECT_EQ(ExactChecker::first_mismatching_lane(aNaN, Vector(aNaN)), LANES);
}

TEST_F(LlvmLibcMathvecCheckerTest, EveryLane) {
  for (size_t lane = 0; lane < LANES; ++lane) {
    Vector actual(1.0f);
    actual[lane] = 2.0f;
    EXPECT_EQ(ExactChecker::first_mismatching_lane(1.0f, actual), lane);
  }
}

TEST_F(LlvmLibcMathvecCheckerTest, TolerantEveryLane) {
  Vector actual(FPBits(0x3f800001U).get_val());
  EXPECT_EQ(TolerantChecker::first_mismatching_lane(1.0f, actual), LANES);
  for (size_t lane = 0; lane < LANES; ++lane) {
    actual[lane] = FPBits(0x3f800002U).get_val();
    EXPECT_EQ(TolerantChecker::first_mismatching_lane(1.0f, actual), lane);
    actual[lane] = FPBits(0x3f800001U).get_val();
  }
}

TEST_F(LlvmLibcMathvecCheckerTest, SignedZero) {
  Vector actual(0.0f);
  for (size_t lane = 0; lane < LANES; ++lane) {
    actual[lane] = -0.0f;
    EXPECT_EQ(ExactChecker::first_mismatching_lane(0.0f, actual), lane);
    actual[lane] = 0.0f;
  }
}
