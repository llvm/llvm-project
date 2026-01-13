//===-- Unittests for cpp::tuple ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/tuple.h"
#include <stddef.h>

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::cpp;

TEST(LlvmLibcTupleTest, Construction) {
  tuple<int, double> t(42, 3.14);
  EXPECT_EQ(get<0>(t), 42);
  EXPECT_FP_EQ(get<1>(t), 3.14);
}

TEST(LlvmLibcTupleTest, MakeTuple) {
  auto t = make_tuple(1, 2.5, 'x');
  EXPECT_EQ(get<0>(t), 1);
  EXPECT_FP_EQ(get<1>(t), 2.5);
  EXPECT_EQ(get<2>(t), 'x');
}

TEST(LlvmLibcTupleTest, TieAssignment) {
  int a = 0;
  double b = 0;
  char c = 0;
  auto t = make_tuple(7, 8.5, 'y');
  tie(a, b, c) = t;
  EXPECT_EQ(a, 7);
  EXPECT_FP_EQ(b, 8.5);
  EXPECT_EQ(c, 'y');
}

TEST(LlvmLibcTupleTest, StructuredBindings) {
  auto t = make_tuple(7, 8.5, 'y');
  auto [x, y, z] = t;
  EXPECT_EQ(x, 7);
  EXPECT_FP_EQ(y, 8.5);
  EXPECT_EQ(z, 'y');
}

TEST(LlvmLibcTupleTest, TupleCat) {
  tuple<int, double> t(42, 3.14);
  auto t1 = make_tuple(1, 2.5, 'x');
  auto t2 = tuple_cat(t, t1);
  EXPECT_EQ(get<0>(t2), 42);
  EXPECT_FP_EQ(get<1>(t2), 3.14);
  EXPECT_EQ(get<2>(t2), 1);
  EXPECT_FP_EQ(get<3>(t2), 2.5);
  EXPECT_EQ(get<4>(t2), 'x');
}

TEST(LlvmLibcTupleTest, ConstTuple) {
  const auto t = make_tuple(100, 200.5);
  EXPECT_EQ(get<0>(t), 100);
  EXPECT_FP_EQ(get<1>(t), 200.5);
}

TEST(LlvmLibcTupleTest, RvalueAssignment) {
  auto t = make_tuple(0, 0.0);
  t = make_tuple(9, 9.5);
  EXPECT_EQ(get<0>(t), 9);
  EXPECT_FP_EQ(get<1>(t), 9.5);
}
