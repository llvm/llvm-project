//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getloadavg.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/getloadavg.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcGetloadavgTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetloadavgTest, ZeroElements) {
  double samples[3] = {-1.0, -1.0, -1.0};
  EXPECT_THAT(LIBC_NAMESPACE::getloadavg(samples, 0), Succeeds(0));
  EXPECT_TRUE(samples[0] == -1.0);
  EXPECT_TRUE(samples[1] == -1.0);
  EXPECT_TRUE(samples[2] == -1.0);
}

TEST_F(LlvmLibcGetloadavgTest, NullptrZeroElements) {
  EXPECT_THAT(LIBC_NAMESPACE::getloadavg(nullptr, 0), Succeeds(0));
}

TEST_F(LlvmLibcGetloadavgTest, NegativeElements) {
  double samples[3] = {-1.0, -1.0, -1.0};
  EXPECT_THAT(LIBC_NAMESPACE::getloadavg(samples, -1), Succeeds(0));
  EXPECT_TRUE(samples[0] == -1.0);
  EXPECT_TRUE(samples[1] == -1.0);
  EXPECT_TRUE(samples[2] == -1.0);
}

TEST_F(LlvmLibcGetloadavgTest, ValidSamples) {
  for (int n = 1; n <= 3; ++n) {
    double samples[3] = {-1.0, -1.0, -1.0};
    EXPECT_THAT(LIBC_NAMESPACE::getloadavg(samples, n), Succeeds(n));

    for (int i = 0; i < n; ++i) {
      EXPECT_TRUE(samples[i] >= 0.0);
    }
    for (int i = n; i < 3; ++i) {
      EXPECT_TRUE(samples[i] == -1.0);
    }
  }
}

TEST_F(LlvmLibcGetloadavgTest, ClampedToThree) {
  double samples[5] = {-1.0, -1.0, -1.0, -1.0, -1.0};
  EXPECT_THAT(LIBC_NAMESPACE::getloadavg(samples, 5), Succeeds(3));

  EXPECT_TRUE(samples[0] >= 0.0);
  EXPECT_TRUE(samples[1] >= 0.0);
  EXPECT_TRUE(samples[2] >= 0.0);
  EXPECT_TRUE(samples[3] == -1.0);
  EXPECT_TRUE(samples[4] == -1.0);
}
