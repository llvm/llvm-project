//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// void swap(flat_map& y) noexcept;

#include <flat_map>
#include <cassert>
#include <deque>

#include "MoveOnly.h"
#include "min_allocator.h"
#include "test_macros.h"
#include "../helpers.h"

// test noexcept

template <class T>
concept NoExceptMemberSwap = requires(T t1, T t2) {
  { t1.swap(t2) } noexcept;
};

static_assert(NoExceptMemberSwap<std::flat_map<int, int>>);
static_assert(
    NoExceptMemberSwap<std::flat_map<int, int, std::less<int>, ThrowOnMoveContainer<int>, ThrowOnMoveContainer<int>>>);

int main(int, char**) {
  using V = std::pair<const int, double>;
  {
    using M = std::flat_map<int, double>;
    {
      M m1;
      M m2;
      M m1_save = m1;
      M m2_save = m2;
      m1.swap(m2);
      assert(m1 == m2_save);
      assert(m2 == m1_save);
    }
    {
      V ar2[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};
      M m1;
      M m2(ar2, ar2 + sizeof(ar2) / sizeof(ar2[0]));
      M m1_save = m1;
      M m2_save = m2;
      m1.swap(m2);
      assert(m1 == m2_save);
      assert(m2 == m1_save);
    }
    {
      V ar1[] = {V(1, 1), V(2, 2), V(3, 3), V(4, 4)};
      M m1(ar1, ar1 + sizeof(ar1) / sizeof(ar1[0]));
      M m2;
      M m1_save = m1;
      M m2_save = m2;
      m1.swap(m2);
      assert(m1 == m2_save);
      assert(m2 == m1_save);
    }
    {
      V ar1[] = {V(1, 1), V(2, 2), V(3, 3), V(4, 4)};
      V ar2[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};
      M m1(ar1, ar1 + sizeof(ar1) / sizeof(ar1[0]));
      M m2(ar2, ar2 + sizeof(ar2) / sizeof(ar2[0]));
      M m1_save = m1;
      M m2_save = m2;
      m1.swap(m2);
      assert(m1 == m2_save);
      assert(m2 == m1_save);
    }
  }
  {
    using M =
        std::flat_map<int,
                      double,
                      std::less<int>,
                      std::vector<int, min_allocator<int>>,
                      std::vector<double, min_allocator<double>>>;
    {
      M m1;
      M m2;
      M m1_save = m1;
      M m2_save = m2;
      m1.swap(m2);
      assert(m1 == m2_save);
      assert(m2 == m1_save);
    }
    {
      V ar2[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};
      M m1;
      M m2(ar2, ar2 + sizeof(ar2) / sizeof(ar2[0]));
      M m1_save = m1;
      M m2_save = m2;
      m1.swap(m2);
      assert(m1 == m2_save);
      assert(m2 == m1_save);
    }
    {
      V ar1[] = {V(1, 1), V(2, 2), V(3, 3), V(4, 4)};
      M m1(ar1, ar1 + sizeof(ar1) / sizeof(ar1[0]));
      M m2;
      M m1_save = m1;
      M m2_save = m2;
      m1.swap(m2);
      assert(m1 == m2_save);
      assert(m2 == m1_save);
    }
    {
      V ar1[] = {V(1, 1), V(2, 2), V(3, 3), V(4, 4)};
      V ar2[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};
      M m1(ar1, ar1 + sizeof(ar1) / sizeof(ar1[0]));
      M m2(ar2, ar2 + sizeof(ar2) / sizeof(ar2[0]));
      M m1_save = m1;
      M m2_save = m2;
      m1.swap(m2);
      assert(m1 == m2_save);
      assert(m2 == m1_save);
    }
  }

  {
    auto swap_func = [](auto& m1, auto& m2) { m1.swap(m2); };
    test_swap_exception_guarantee(swap_func);
  }

  return 0;
}
