//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for swap.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/utility/swap.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

TEST(LlvmLibcSwapTest, Primitives) {
  int a = 10;
  int b = 20;
  swap(a, b);
  EXPECT_EQ(a, 20);
  EXPECT_EQ(b, 10);

  char c1 = 'x';
  char c2 = 'y';
  swap(c1, c2);
  EXPECT_EQ(c1, 'y');
  EXPECT_EQ(c2, 'x');

  uint64_t u1 = 0x123456789abcdef0;
  uint64_t u2 = 0x0fedcba987654321;
  swap(u1, u2);
  EXPECT_EQ(u1, uint64_t(0x0fedcba987654321));
  EXPECT_EQ(u2, uint64_t(0x123456789abcdef0));
}

struct MoveTracker {
  int val = 0;
  int moves = 0;
  int copies = 0;

  constexpr MoveTracker(int v) : val(v) {}
  constexpr MoveTracker(const MoveTracker &o)
      : val(o.val), moves(o.moves), copies(o.copies + 1) {}
  constexpr MoveTracker(MoveTracker &&o)
      : val(o.val), moves(o.moves + 1), copies(o.copies) {
    o.val = 0;
  }
  constexpr MoveTracker &operator=(const MoveTracker &o) {
    val = o.val;
    copies = o.copies + 1;
    return *this;
  }
  constexpr MoveTracker &operator=(MoveTracker &&o) {
    val = o.val;
    moves = o.moves + 1;
    o.val = 0;
    return *this;
  }
};

TEST(LlvmLibcSwapTest, MoveSemantics) {
  MoveTracker t1(100);
  MoveTracker t2(200);

  swap(t1, t2);

  EXPECT_EQ(t1.val, 200);
  EXPECT_EQ(t2.val, 100);
  // swap should perform 3 moves and 0 copies
  EXPECT_EQ(t1.copies, 0);
  EXPECT_EQ(t2.copies, 0);
  EXPECT_EQ(t1.moves + t2.moves, 3);
}

struct MoveOnly {
  int val;
  constexpr MoveOnly(int v) : val(v) {}
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly &operator=(const MoveOnly &) = delete;
  constexpr MoveOnly(MoveOnly &&o) : val(o.val) { o.val = 0; }
  constexpr MoveOnly &operator=(MoveOnly &&o) {
    val = o.val;
    o.val = 0;
    return *this;
  }
};

TEST(LlvmLibcSwapTest, MoveOnlyType) {
  MoveOnly m1(1);
  MoveOnly m2(2);
  swap(m1, m2);
  EXPECT_EQ(m1.val, 2);
  EXPECT_EQ(m2.val, 1);
}

TEST(LlvmLibcSwapTest, ArraySwap) {
  int arr1[3] = {1, 2, 3};
  int arr2[3] = {4, 5, 6};

  swap(arr1, arr2);

  EXPECT_EQ(arr1[0], 4);
  EXPECT_EQ(arr1[1], 5);
  EXPECT_EQ(arr1[2], 6);
  EXPECT_EQ(arr2[0], 1);
  EXPECT_EQ(arr2[1], 2);
  EXPECT_EQ(arr2[2], 3);
}

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL
