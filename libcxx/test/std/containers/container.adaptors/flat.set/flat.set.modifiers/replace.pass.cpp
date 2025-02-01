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

using Set = std::flat_set<int, int>;
static_assert(CanReplace<Set, std::vector<int>>);
static_assert(!CanReplace<Set, const std::vector<int>&>);

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;

  M m                   = M({1, 2, 3});
  KeyContainer new_keys = {7, 8};
  auto expected_keys    = new_keys;
  m.replace(std::move(new_keys));
  assert(m.size() == 2);
  assert(std::ranges::equal(m, expected_keys));
}

int main(int, char**) {
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();

  {
#ifndef TEST_HAS_NO_EXCEPTIONS
    using KeyContainer = ThrowOnMoveContainer<int>;
    using M            = std::flat_set<int, std::ranges::less, KeyContainer>;

    M m;
    m.emplace(1);
    m.emplace(2);
    try {
      KeyContainer new_keys{3, 4};
      m.replace(std::move(new_keys));
      assert(false);
    } catch (int) {
      check_invariant(m);
      // In libc++, we clear the map
      LIBCPP_ASSERT(m.size() == 0);
    }
#endif
  }
  return 0;
}
