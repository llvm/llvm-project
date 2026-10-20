//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// class flat_multimap

// void clear() noexcept;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

// test noexcept

template <class T>
concept NoExceptClear = requires(T t) {
  { t.clear() } noexcept;
};

static_assert(NoExceptClear<std::flat_multimap<int, int>>);
#ifndef TEST_HAS_NO_EXCEPTIONS
static_assert(
    NoExceptClear<std::flat_multimap<int, int, std::less<int>, ThrowOnMoveContainer<int>, ThrowOnMoveContainer<int>>>);
#endif

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  M m = {{5, 2}, {2, 1}, {2, 3}, {2, 1}, {5, 0}};
  assert(m.size() == 5);
  ASSERT_NOEXCEPT(m.clear());
  ASSERT_SAME_TYPE(decltype(m.clear()), void);
  m.clear();
  assert(m.size() == 0);
}

constexpr bool test() {
  test<std::vector<int>, std::vector<int>>();
  test<std::vector<int>, std::vector<double>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();
  test<std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>();
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
