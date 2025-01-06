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
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
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
#ifndef TEST_HAS_NO_EXCEPTIONS
static_assert(
    NoExceptMemberSwap<std::flat_map<int, int, std::less<int>, ThrowOnMoveContainer<int>, ThrowOnMoveContainer<int>>>);
#endif

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using V     = std::pair<const Key, Value>;
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

int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  return 0;
}
