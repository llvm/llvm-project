//===-- flang/unittests/Common/FastIntSetTest.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/fast-int-set.h"
#include <optional>

TEST(FastIntSetTests, Sanity) {
  static constexpr int N{100};
  Fortran::common::FastIntSet<N> set;

  ASSERT_FALSE(set.IsValidValue(-1));
  ASSERT_TRUE(set.IsValidValue(0));
  ASSERT_TRUE(set.IsValidValue(N - 1));
  ASSERT_FALSE(set.IsValidValue(N));
  ASSERT_TRUE(set.IsEmpty());
  ASSERT_EQ(set.size(), 0);
  ASSERT_FALSE(set.Contains(0));
  ASSERT_FALSE(set.Contains(N - 1));

  ASSERT_TRUE(set.Add(0));
  ASSERT_FALSE(set.IsEmpty());
  ASSERT_EQ(set.size(), 1);
  ASSERT_TRUE(set.Contains(0));

  ASSERT_TRUE(set.Add(0)); // duplicate
  ASSERT_EQ(set.size(), 1);
  ASSERT_TRUE(set.Contains(0));

  ASSERT_TRUE(set.Remove(0));
  ASSERT_TRUE(set.IsEmpty());
  ASSERT_EQ(set.size(), 0);
  ASSERT_FALSE(set.Contains(0));

  ASSERT_FALSE(set.Add(N));
  ASSERT_TRUE(set.IsEmpty());
  ASSERT_EQ(set.size(), 0);
  ASSERT_FALSE(set.Contains(N));

  ASSERT_TRUE(set.Add(N - 1));
  ASSERT_FALSE(set.IsEmpty());
  ASSERT_EQ(set.size(), 1);
  ASSERT_TRUE(set.Contains(N - 1));

  std::optional<int> x;
  x = set.PopValue();
  ASSERT_TRUE(x.has_value());
  ASSERT_EQ(*x, N - 1);
  ASSERT_TRUE(set.IsEmpty());
  ASSERT_EQ(set.size(), 0);

  x = set.PopValue();
  ASSERT_FALSE(x.has_value());

  for (int j{0}; j < N; ++j) {
    ASSERT_TRUE(set.Add(j)) << j;
  }
  ASSERT_FALSE(set.IsEmpty());
  ASSERT_EQ(set.size(), N);
  for (int j{0}; j < N; ++j) {
    ASSERT_TRUE(set.Contains(j)) << j;
  }

  for (int j{0}; j < N; ++j) {
    ASSERT_TRUE(set.Remove(j)) << j;
    ASSERT_EQ(set.size(), N - j - 1) << j;
    ASSERT_FALSE(set.Contains(j)) << j;
  }

  ASSERT_TRUE(set.IsEmpty());
  ASSERT_EQ(set.size(), 0);

  for (int j{N - 1}; j >= 0; --j) {
    ASSERT_TRUE(set.Add(j)) << j;
  }
  for (int j{0}; j < N; j++) {
    x = set.PopValue();
    ASSERT_TRUE(x.has_value());
    ASSERT_EQ(*x, j) << j;
  }
  ASSERT_TRUE(set.IsEmpty());
  ASSERT_EQ(set.size(), 0);

  for (int j{0}; j < N; j++) {
    ASSERT_TRUE(set.Add(j)) << j;
  }
  ASSERT_FALSE(set.IsEmpty());
  ASSERT_EQ(set.size(), N);
  for (int j{0}; j < N; j += 2) {
    ASSERT_TRUE(set.Remove(j)) << j;
  }
  ASSERT_FALSE(set.IsEmpty());
  ASSERT_EQ(set.size(), N / 2);
  for (int j{0}; j < N; j++) {
    ASSERT_EQ(set.Contains(j), (j & 1) == 1);
  }

  set.Clear();
  ASSERT_TRUE(set.IsEmpty());
}
