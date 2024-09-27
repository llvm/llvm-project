//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

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

#include "test_macros.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    using M                                                      = std::flat_map<int, char>;
    const M m                                                    = {{4, 'a'}, {2, 'b'}, {3, 'c'}};
    std::same_as<const std::vector<int>&> decltype(auto) keys    = m.keys();
    std::same_as<const std::vector<char>&> decltype(auto) values = m.values();

    // noexcept
    static_assert(noexcept(m.keys()));
    static_assert(noexcept(m.values()));

    auto expected_keys   = {2, 3, 4};
    auto expected_values = {'b', 'c', 'a'};
    assert(std::ranges::equal(keys, expected_keys));
    assert(std::ranges::equal(values, expected_values));
  }

  {
    using KeyContainer   = std::deque<double>;
    using ValueContainer = std::vector<int, test_allocator<int>>;
    using M              = std::flat_map<double, int, std::less<>, KeyContainer, ValueContainer>;
    const M m            = {{1.0, 1}, {4.0, 4}, {2.0, 2}};
    std::same_as<const KeyContainer&> decltype(auto) keys     = m.keys();
    std::same_as<const ValueContainer&> decltype(auto) values = m.values();

    // noexcept
    static_assert(noexcept(m.keys()));
    static_assert(noexcept(m.values()));

    auto expected_keys   = {1.0, 2.0, 4.0};
    auto expected_values = {1, 2, 4};
    assert(std::ranges::equal(keys, expected_keys));
    assert(std::ranges::equal(values, expected_values));
  }

  return 0;
}
