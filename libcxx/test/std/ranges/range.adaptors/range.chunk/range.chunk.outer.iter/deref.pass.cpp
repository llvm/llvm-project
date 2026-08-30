//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models only input_range:
//     constexpr value_type outer_iterator::operator*() const;
//     constexpr inner_iterator outer_iterator::value_type::begin() const noexcept;
//     constexpr default_sentinel_t outer_iterator::value_type::end() const noexcept;

#include <cassert>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Test `constexpr value_type outer_iterator::operator*() const`
  {
    std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked =
      std::ranges::subrange<Iterator, Sentinel>(
        Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))) |
        std::views::chunk(3);

    static_assert(
        std::same_as<decltype(*chunked.begin()), typename std::ranges::iterator_t<decltype(chunked)>::value_type>);
    static_assert(std::ranges::input_range<typename std::ranges::iterator_t<decltype(chunked)>::value_type>);
  }

  // Test `constexpr inner_iterator outer_iterator::value_type::begin() const noexcept`
  {
    std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked =
      std::ranges::subrange<Iterator, Sentinel>(
        Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))) |
        std::views::chunk(3);

    /*chunk_view::__outer_iterator::value_type*/ std::ranges::input_range auto inner = *chunked.begin();
    assert(*inner.begin() == *vector.begin());
    static_assert(noexcept(inner.begin()));
  }

  // Test `constexpr default_sentinel_t outer_iterator::value_type::end() const noexcept`
  {
    std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked =
      std::ranges::subrange<Iterator, Sentinel>(
        Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))) |
        std::views::chunk(3);

    /*chunk_view::__outer_iterator::value_type*/ std::ranges::input_range auto inner = *chunked.begin();
    [[maybe_unused]] std::same_as<std::default_sentinel_t> auto it                   = inner.end();
    static_assert(noexcept(inner.end()));
  }

  return true;
}

int main(int, char**) {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>();

  static_assert(test<cpp17_input_iterator<int*>>());
  static_assert(test<cpp20_input_iterator<int*>>());
  static_assert(test<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>());

  return 0;
}
