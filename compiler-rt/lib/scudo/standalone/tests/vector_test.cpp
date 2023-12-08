//===-- vector_test.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "vector.h"

TEST(ScudoVectorTest, Basic) {
  scudo::Vector<int> V;
  EXPECT_EQ(V.size(), 0U);
  V.push_back(42);
  EXPECT_EQ(V.size(), 1U);
  EXPECT_EQ(V[0], 42);
  V.push_back(43);
  EXPECT_EQ(V.size(), 2U);
  EXPECT_EQ(V[0], 42);
  EXPECT_EQ(V[1], 43);
}

TEST(ScudoVectorTest, Stride) {
  scudo::Vector<scudo::uptr> V;
  for (scudo::uptr I = 0; I < 1000; I++) {
    V.push_back(I);
    EXPECT_EQ(V.size(), I + 1U);
    EXPECT_EQ(V[I], I);
  }
  for (scudo::uptr I = 0; I < 1000; I++)
    EXPECT_EQ(V[I], I);
}

TEST(ScudoVectorTest, ResizeReduction) {
  scudo::Vector<int> V;
  V.push_back(0);
  V.push_back(0);
  EXPECT_EQ(V.size(), 2U);
  V.resize(1);
  EXPECT_EQ(V.size(), 1U);
}
