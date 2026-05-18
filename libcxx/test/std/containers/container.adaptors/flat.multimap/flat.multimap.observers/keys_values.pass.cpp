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

// const key_container_type& keys() const noexcept
// const mapped_container_type& values() const noexcept

#include <algorithm>
#include <cassert>
#include <flat_map>
#include <functional>
#include <utility>
#include <vector>
#include <deque>
#include <string>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "../helpers.h"

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  const M m                                                 = {{4, 'a'}, {2, 'b'}, {2, 'e'}, {3, 'c'}};
  std::same_as<const KeyContainer&> decltype(auto) keys     = m.keys();
  std::same_as<const ValueContainer&> decltype(auto) values = m.values();

  // noexcept
  static_assert(noexcept(m.keys()));
  static_assert(noexcept(m.values()));

  auto expected_keys = {2, 2, 3, 4};
  assert(std::ranges::equal(keys, expected_keys));
  check_possible_values(
      values,
      std::vector<std::vector<char>>{
          {'b', 'e'},
          {'b', 'e'},
          {'c'},
          {'a'},
      });
}

constexpr bool test() {
  test<std::vector<int>, std::vector<char>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>, std::vector<char>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<char>>();
  test<std::vector<int, min_allocator<int>>, std::vector<char, min_allocator<char>>>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
