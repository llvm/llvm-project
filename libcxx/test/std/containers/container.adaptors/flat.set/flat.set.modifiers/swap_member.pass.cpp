//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// void swap(flat_set& y) noexcept;

#include <flat_set>
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

static_assert(NoExceptMemberSwap<std::flat_set<int>>);
#ifndef TEST_HAS_NO_EXCEPTIONS
static_assert(NoExceptMemberSwap<std::flat_set<int, std::less<int>, ThrowOnMoveContainer<int>>>);
#endif

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;
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
    int ar2[] = {5, 6, 7, 8, 9, 10, 11, 12};
    M m1;
    M m2(ar2, ar2 + sizeof(ar2) / sizeof(ar2[0]));
    M m1_save = m1;
    M m2_save = m2;
    m1.swap(m2);
    assert(m1 == m2_save);
    assert(m2 == m1_save);
  }
  {
    int ar1[] = {1, 2, 3, 4};
    M m1(ar1, ar1 + sizeof(ar1) / sizeof(ar1[0]));
    M m2;
    M m1_save = m1;
    M m2_save = m2;
    m1.swap(m2);
    assert(m1 == m2_save);
    assert(m2 == m1_save);
  }
  {
    int ar1[] = {1, 2, 3, 4};
    int ar2[] = {5, 6, 7, 8, 9, 10, 11, 12};
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
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();

  return 0;
}
