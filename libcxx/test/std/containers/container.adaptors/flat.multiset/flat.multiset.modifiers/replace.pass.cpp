//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// void replace(container_type&& key_cont);

#include <algorithm>
#include <deque>
#include <concepts>
#include <flat_set>
#include <functional>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class T, class... Args>
concept CanReplace = requires(T t, Args&&... args) { t.replace(std::forward<Args>(args)...); };

using Set = std::flat_multiset<int, int>;
static_assert(CanReplace<Set, std::vector<int>>);
static_assert(!CanReplace<Set, const std::vector<int>&>);

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_multiset<Key, std::less<Key>, KeyContainer>;
  {
    // was empty
    M m;
    KeyContainer new_keys = {7, 7, 8};
    auto expected_keys    = new_keys;
    m.replace(std::move(new_keys));
    assert(m.size() == 3);
    assert(std::ranges::equal(m, expected_keys));
  }
  {
    M m                   = M({1, 1, 2, 2, 3});
    KeyContainer new_keys = {7, 7, 8, 8};
    auto expected_keys    = new_keys;
    m.replace(std::move(new_keys));
    assert(m.size() == 4);
    assert(std::ranges::equal(m, expected_keys));
  }
}

constexpr bool test() {
  test_one<std::vector<int>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();

  return true;
}

void test_exception() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using KeyContainer = ThrowOnMoveContainer<int>;
  using M            = std::flat_multiset<int, std::ranges::less, KeyContainer>;

  M m;
  m.emplace(1);
  m.emplace(2);
  try {
    KeyContainer new_keys{3, 4};
    m.replace(std::move(new_keys));
    assert(false);
  } catch (int) {
    check_invariant(m);
    // In libc++, we clear the set
    LIBCPP_ASSERT(m.size() == 0);
  }
#endif
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  test_exception();

  return 0;
}
