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

// void replace(key_container_type&& key_cont, mapped_container_type&& mapped_cont);

#include <algorithm>
#include <deque>
#include <concepts>
#include <flat_map>
#include <functional>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class T, class... Args>
concept CanReplace = requires(T t, Args&&... args) { t.replace(std::forward<Args>(args)...); };

using Map = std::flat_multimap<int, int>;
static_assert(CanReplace<Map, std::vector<int>, std::vector<int>>);
static_assert(!CanReplace<Map, const std::vector<int>&, std::vector<int>>);
static_assert(!CanReplace<Map, std::vector<int>, const std::vector<int>&>);
static_assert(!CanReplace<Map, const std::vector<int>&, const std::vector<int>&>);

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  M m                       = M({1, 1, 3}, {4, 5, 6});
  KeyContainer new_keys     = {7, 7};
  ValueContainer new_values = {9, 10};
  auto expected_keys        = new_keys;
  auto expected_values      = new_values;
  m.replace(std::move(new_keys), std::move(new_values));
  assert(m.size() == 2);
  assert(std::ranges::equal(m.keys(), expected_keys));
  assert(std::ranges::equal(m.values(), expected_values));
}

constexpr bool test() {
  test<std::vector<int>, std::vector<double>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  if (!TEST_IS_CONSTANT_EVALUATED) {
#ifndef TEST_HAS_NO_EXCEPTIONS
    using KeyContainer   = std::vector<int>;
    using ValueContainer = ThrowOnMoveContainer<int>;
    using M              = std::flat_multimap<int, int, std::ranges::less, KeyContainer, ValueContainer>;

    M m;
    m.emplace(1, 1);
    m.emplace(2, 2);
    try {
      KeyContainer new_keys{3, 4};
      ValueContainer new_values{5, 6};
      m.replace(std::move(new_keys), std::move(new_values));
      assert(false);
    } catch (int) {
      check_invariant(m);
      // In libc++, we clear the map
      LIBCPP_ASSERT(m.size() == 0);
    }
#endif
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
