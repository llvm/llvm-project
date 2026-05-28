//===-- tsan_percent_test.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "gtest/gtest.h"
#include "tsan_adaptive_delay.h"

namespace __tsan {

TEST(Percent, DefaultedObject) {
  Percent defaulted;
  EXPECT_FALSE(defaulted.IsValid());
}

TEST(Percent, FromPct) {
  Percent p0 = Percent::FromPct(0);
  Percent p50 = Percent::FromPct(50);
  Percent p100 = Percent::FromPct(100);
  Percent p150 = Percent::FromPct(150);

  EXPECT_TRUE(p0.IsValid());
  EXPECT_TRUE(p50.IsValid());
  EXPECT_TRUE(p100.IsValid());
  EXPECT_TRUE(p150.IsValid());

  EXPECT_EQ(p0, p0);
  EXPECT_NE(p0, p50);
  EXPECT_NE(p50, p100);

  EXPECT_EQ(p0.GetBasisPoints(), 0);
  EXPECT_EQ(p50.GetBasisPoints(), 5000);
  EXPECT_EQ(p100.GetBasisPoints(), 10000);
  EXPECT_EQ(p150.GetBasisPoints(), 15000);

  EXPECT_EQ(p0.GetPct(), 0);
  EXPECT_EQ(p50.GetPct(), 50);
  EXPECT_EQ(p100.GetPct(), 100);
  EXPECT_EQ(p150.GetPct(), 150);
}

TEST(Percent, FromRatio) {
  Percent half = Percent::FromRatio(1, 2);
  Percent expected_half = Percent::FromPct(50);
  EXPECT_TRUE(half.IsValid());
  EXPECT_EQ(half, expected_half);

  Percent quarter = Percent::FromRatio(1, 4);
  Percent expected_quarter = Percent::FromPct(25);
  EXPECT_EQ(quarter, expected_quarter);

  Percent full = Percent::FromRatio(100, 100);
  Percent expected_full = Percent::FromPct(100);
  EXPECT_EQ(full, expected_full);

  Percent div_zero = Percent::FromRatio(50, 0);
  EXPECT_FALSE(div_zero.IsValid());
}

TEST(Percent, Comparisons) {
  Percent low = Percent::FromPct(20);
  Percent p20 = Percent::FromPct(20);
  Percent mid = Percent::FromPct(50);
  Percent high = Percent::FromPct(80);

  EXPECT_TRUE(low == p20);
  EXPECT_FALSE(low != p20);
  EXPECT_FALSE(low == mid);
  EXPECT_TRUE(low != mid);

  EXPECT_TRUE(low < mid);
  EXPECT_TRUE(mid < high);
  EXPECT_FALSE(high < low);

  EXPECT_TRUE(high > mid);
  EXPECT_TRUE(mid > low);
  EXPECT_FALSE(low > high);

  EXPECT_TRUE(low <= mid);
  EXPECT_TRUE(low <= low);

  EXPECT_TRUE(high >= mid);
  EXPECT_TRUE(high >= high);
}

TEST(Percent, Subtraction) {
  Percent a = Percent::FromPct(75);
  Percent b = Percent::FromPct(25);
  Percent result = a - b;

  Percent expected = Percent::FromPct(50);
  EXPECT_TRUE(result.IsValid());
  EXPECT_EQ(result, expected);

  // Underflow
  Percent low = Percent::FromPct(20);
  Percent high = Percent::FromPct(80);
  Percent underflow = low - high;
  EXPECT_FALSE(underflow.IsValid());

  Percent result_invalid = underflow - low;
  EXPECT_FALSE(result_invalid.IsValid());
}

TEST(Percent, Division) {
  Percent numerator = Percent::FromPct(100);
  Percent denominator = Percent::FromPct(50);
  Percent result = numerator / denominator;

  Percent expected = Percent::FromPct(200);
  EXPECT_TRUE(result.IsValid());
  EXPECT_EQ(result, expected);

  Percent zero = Percent::FromPct(0);
  Percent non_zero = Percent::FromPct(50);
  Percent div_zero_result = non_zero / zero;
  EXPECT_FALSE(div_zero_result.IsValid());

  Percent invalid = Percent::FromRatio(10, 0);
  Percent valid = Percent::FromPct(50);
  Percent result_invalid = valid / invalid;
  EXPECT_FALSE(result_invalid.IsValid());
}

TEST(Percent, RandomCheck) {
  unsigned int seed = 0;

  Percent p0 = Percent::FromPct(0);
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(p0.RandomCheck(&seed));
  }

  Percent p50 = Percent::FromPct(50);
  for (int i = 0; i < 100; ++i) {
    p50.RandomCheck(&seed);
    // No verification since we cannot guarantee the random result.
    // Just verify the code does not crash...
  }

  Percent p150 = Percent::FromPct(150);
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(p150.RandomCheck(&seed));
  }
}

}  // namespace __tsan
