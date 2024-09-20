//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// containers extract() &&;

#include <algorithm>
#include <concepts>
#include <flat_map>
#include <functional>

#include "../helpers.h"
#include "test_macros.h"

template <class T>
concept CanExtract = requires(T&& t) { std::forward<T>(t).extract(); };

static_assert(CanExtract<std::flat_map<int, int>&&>);
static_assert(!CanExtract<std::flat_map<int, int>&>);
static_assert(!CanExtract<std::flat_map<int, int> const&>);
static_assert(!CanExtract<std::flat_map<int, int> const&&>);

int main(int, char**) {
  {
    using M                                     = std::flat_map<int, int>;
    M m                                         = M({1, 2, 3}, {4, 5, 6});
    std::same_as<M::containers> auto containers = std::move(m).extract();
    auto expected_keys                          = {1, 2, 3};
    auto expected_values                        = {4, 5, 6};
    assert(std::ranges::equal(containers.keys, expected_keys));
    assert(std::ranges::equal(containers.values, expected_values));
    LIBCPP_ASSERT(m.empty());
    LIBCPP_ASSERT(m.keys().size() == 0);
    LIBCPP_ASSERT(m.values().size() == 0);
  }
  {
    // extracted object maintains invariant if one of underlying container does not clear after move
    using M = std::flat_map<int, int, std::less<>, std::vector<int>, CopyOnlyVector<int>>;
    M m     = M({1, 2, 3}, {1, 2, 3});
    std::same_as<M::containers> auto containers = std::move(m).extract();
    assert(containers.keys.size() == 3);
    assert(containers.values.size() == 3);
    LIBCPP_ASSERT(m.empty());
    LIBCPP_ASSERT(m.keys().size() == 0);
    LIBCPP_ASSERT(m.values().size() == 0);
  }

  return 0;
}
