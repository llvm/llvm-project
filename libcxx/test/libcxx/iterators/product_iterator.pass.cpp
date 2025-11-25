//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// gcc 15 does not seem to recognize the __product_iterator_traits specializations
// UNSUPPORTED: gcc

#include <flat_map>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"

constexpr bool test() {
  {
    // Test that the __get_iterator_element can handle a non-copyable iterator
    int Date[] = {1, 2, 3, 4};
    cpp20_input_iterator<int*> iter(Date);
    sentinel_wrapper<cpp20_input_iterator<int*>> sent{cpp20_input_iterator<int*>(Date + 4)};
    std::ranges::subrange r1(std::move(iter), std::move(sent));
    auto v  = std::views::zip(std::move(r1), std::views::iota(0, 4));
    auto it = v.begin();

    using Iter = decltype(it);

    static_assert(!std::is_copy_constructible_v<Iter>);

    static_assert(std::__product_iterator_traits<Iter>::__size == 2);
    std::same_as<cpp20_input_iterator<int*>&> decltype(auto) it1 =
        std::__product_iterator_traits<Iter>::__get_iterator_element<0>(it);

    assert(*it1 == 1);
  }
  if (!std::is_constant_evaluated()) {
    // Test __make_product_iterator
    using M = std::flat_map<int, int>;
    M m{{1, 1}, {2, 2}, {3, 3}};
    using Iter         = std::ranges::iterator_t<const M>;
    const auto& keys   = m.keys();
    const auto& values = m.values();

    auto it_keys   = std::ranges::begin(keys);
    auto it_values = std::ranges::begin(values);

    auto it = std::__product_iterator_traits<Iter>::__make_product_iterator(it_keys, it_values);
    assert(it->first == 1);
    assert(it->second == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
