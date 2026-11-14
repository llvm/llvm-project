//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/simd.h"
#include "test/UnitTest/Test.h"
#include "test/src/mathvec/UnitTestWrappers.h"

namespace {

constexpr size_t LANES =
    LIBC_NAMESPACE::cpp::internal::native_vector_size<float>;
bool seen_control[LANES];

float identity(float x) { return x; }

LIBC_VECTOR_PCS LIBC_NAMESPACE::cpp::simd<float>
record_control(LIBC_NAMESPACE::cpp::simd<float> input) {
  for (size_t lane = 0; lane < LANES; ++lane)
    if (input[lane] == 1.0f)
      seen_control[lane] = true;
  return input;
}

using RecordingOp =
    LIBC_NAMESPACE::testing::mathvec::UnaryOp<float, identity, record_control>;
using LIBC_NAMESPACE::testing::mathvec::wrap_ref;
using LIBC_NAMESPACE::testing::mathvec::wrap_vector;
using LlvmLibcMathvecWrappersTest = LIBC_NAMESPACE::testing::FPTest<float>;

} // namespace

TEST_F(LlvmLibcMathvecWrappersTest, ControlLane) {
  for (size_t control_lane = 0; control_lane < LANES; ++control_lane) {
    auto input =
        LIBC_NAMESPACE::testing::mathvec::internal::make_input<RecordingOp>(
            2.0f, -0.0f, control_lane);
    for (size_t lane = 0; lane < LANES; ++lane)
      EXPECT_FP_EQ(lane == control_lane ? -0.0f : 2.0f, input[lane]);
  }
}

TEST_F(LlvmLibcMathvecWrappersTest, BroadcastAndDefaultLane) {
  auto broadcast = wrap_vector<RecordingOp>(2.0f);
  auto mixed = wrap_vector<RecordingOp>(2.0f, -0.0f);
  for (size_t lane = 0; lane < LANES; ++lane) {
    EXPECT_FP_EQ(2.0f, broadcast[lane]);
    EXPECT_FP_EQ(lane == 0 ? -0.0f : 2.0f, mixed[lane]);
  }
}

TEST_F(LlvmLibcMathvecWrappersTest, ExactVariedCases) {
  for (size_t lane = 0; lane < LANES; ++lane)
    seen_control[lane] = false;
  TEST_VARIED_CASES(2.0f, RecordingOp);
  for (size_t lane = 0; lane < LANES; ++lane)
    EXPECT_TRUE(seen_control[lane]);
}

TEST_F(LlvmLibcMathvecWrappersTest, TolerantVariedCases) {
  for (size_t lane = 0; lane < LANES; ++lane)
    seen_control[lane] = false;
  TEST_VARIED_CASES_TOL(2.0f, RecordingOp, 1);
  for (size_t lane = 0; lane < LANES; ++lane)
    EXPECT_TRUE(seen_control[lane]);
}
