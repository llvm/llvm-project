//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for Vector.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/checked.h"
#include "src/__support/vector.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Checked;
using LIBC_NAMESPACE::Vector;
using LIBC_NAMESPACE::cpp::is_copy_assignable_v;
using LIBC_NAMESPACE::cpp::is_copy_constructible_v;
using LIBC_NAMESPACE::cpp::is_move_assignable_v;
using LIBC_NAMESPACE::cpp::is_move_constructible_v;
using LIBC_NAMESPACE::cpp::move;

// Copying needs to allocate, and the copy operations can't report failures.
static_assert(!is_copy_constructible_v<Vector<int>>);
static_assert(!is_copy_assignable_v<Vector<int>>);
static_assert(is_move_constructible_v<Vector<int>>);
static_assert(is_move_assignable_v<Vector<int>>);

namespace {

// Vector<Point> is instantiated while Point is still incomplete.
struct Point;
struct Polygon {
  Vector<Point> vertices;
};
struct Point {
  int x;
  int y;
};

} // namespace

// Checks the result of a fallible operation, and converts it to bool so it can
// be used with ASSERT_TRUE and friends.
static bool ok(Checked result) { return static_cast<bool>(result); }

template <size_t N>
static void expect_contents(const Vector<int> &v, const int (&expected)[N]) {
  ASSERT_EQ(v.size(), N);
  for (size_t i = 0; i < N; ++i)
    EXPECT_EQ(v[i], expected[i]);
}

TEST(LlvmLibcVectorTest, Empty) {
  Vector<int> v;
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), size_t(0));
  EXPECT_EQ(v.capacity(), size_t(0));
  EXPECT_TRUE(v.begin() == v.end());
}

TEST(LlvmLibcVectorTest, PushAndPop) {
  Vector<int> v;
  for (int i = 0; i < 20; ++i)
    ASSERT_TRUE(ok(v.push_back(i)));
  ASSERT_EQ(v.size(), size_t(20));
  EXPECT_GE(v.capacity(), size_t(20));
  EXPECT_FALSE(v.empty());
  EXPECT_EQ(v.front(), 0);
  EXPECT_EQ(v.back(), 19);
  for (size_t i = 0; i < v.size(); ++i)
    EXPECT_EQ(v[i], static_cast<int>(i));

  for (int i = 19; i >= 0; --i) {
    EXPECT_EQ(v.back(), i);
    v.pop_back();
  }
  EXPECT_TRUE(v.empty());
}

TEST(LlvmLibcVectorTest, Iteration) {
  Vector<int> v;
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(ok(v.push_back(i)));

  int expected = 0;
  for (int &x : v)
    EXPECT_EQ(x, expected++);
  EXPECT_EQ(expected, 4);

  const Vector<int> &cv = v;
  expected = 0;
  for (const int &x : cv)
    EXPECT_EQ(x, expected++);
  EXPECT_EQ(expected, 4);
}

TEST(LlvmLibcVectorTest, PushBackOwnElementWhileGrowing) {
  Vector<int> v;
  ASSERT_TRUE(ok(v.push_back(1)));
  ASSERT_TRUE(ok(v.push_back(2)));
  // Without spare capacity, the push_back needs to grow the storage, which
  // invalidates the reference to the element being appended.
  ASSERT_EQ(v.size(), v.capacity());
  ASSERT_TRUE(ok(v.push_back(v[0])));
  expect_contents(v, {1, 2, 1});
}

TEST(LlvmLibcVectorTest, MoveAndSwap) {
  Vector<int> a;
  for (int i = 0; i < 3; ++i)
    ASSERT_TRUE(ok(a.push_back(i)));
  int *data = a.data();

  Vector<int> b(move(a));
  EXPECT_TRUE(a.empty());
  EXPECT_EQ(a.capacity(), size_t(0));
  EXPECT_EQ(b.size(), size_t(3));
  EXPECT_EQ(b.data(), data);

  Vector<int> c;
  ASSERT_TRUE(ok(c.push_back(2)));
  c = move(b);
  EXPECT_TRUE(b.empty());
  EXPECT_EQ(c.size(), size_t(3));
  EXPECT_EQ(c.data(), data);

  // Self-move-assignment leaves the vector unchanged.
  Vector<int> &alias = c;
  c = move(alias);
  EXPECT_EQ(c.size(), size_t(3));
  EXPECT_EQ(c.data(), data);

  ASSERT_TRUE(ok(a.push_back(4)));
  a.swap(c);
  EXPECT_EQ(a.size(), size_t(3));
  EXPECT_EQ(a.data(), data);
  expect_contents(c, {4});
}

TEST(LlvmLibcVectorTest, IncompleteType) {
  Polygon p;
  ASSERT_TRUE(ok(p.vertices.push_back(Point{1, 2})));
  ASSERT_TRUE(ok(p.vertices.push_back(Point{3, 4})));

  ASSERT_EQ(p.vertices.size(), size_t(2));
  EXPECT_EQ(p.vertices[0].x, 1);
  EXPECT_EQ(p.vertices[0].y, 2);
  EXPECT_EQ(p.vertices[1].x, 3);
  EXPECT_EQ(p.vertices[1].y, 4);
}

#if !defined(NDEBUG) && LIBC_TEST_SUBPROCESS_TESTS

#if defined(LIBC_FULL_BUILD) && !defined(LIBC_COPT_USE_C_ASSERT)
#define EXPECT_ASSERT_FAILS(stmt) EXPECT_EXITS(stmt, 255)
#else
#define EXPECT_ASSERT_FAILS(stmt) EXPECT_DEATH(stmt, WITH_SIGNAL(-1))
#endif

TEST(LlvmLibcVectorDeathTest, UncheckedResultDies) {
  EXPECT_ASSERT_FAILS([] {
    Vector<int> v;
    (void)v.push_back(1);
  });
}

TEST(LlvmLibcVectorDeathTest, OutOfBoundsAccessDies) {
  EXPECT_ASSERT_FAILS([] {
    Vector<int> v;
    (void)v[0];
  });
}

TEST(LlvmLibcVectorDeathTest, PopBackOnEmptyVectorDies) {
  EXPECT_ASSERT_FAILS([] {
    Vector<int> v;
    v.pop_back();
  });
}

#endif // !defined(NDEBUG) && LIBC_TEST_SUBPROCESS_TESTS
