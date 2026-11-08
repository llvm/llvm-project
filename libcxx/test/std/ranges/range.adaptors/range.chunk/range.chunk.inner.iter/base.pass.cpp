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
//     constexpr const iterator_t<V>& base() const&;

#include <cassert>
#include <concepts>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

#include "test_iterators.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr bool test() {
  std::vector<int> vector        = {1, 2, 3, 4};
  std::vector<int> single_vector = {1};

  // Test `constexpr const iterator_t<V>& base() const&`
  {
    // When range is general
    {
      std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked(
          std::ranges::subrange<Iterator, Sentinel>(
              Iterator(vector.data()), Sentinel(Iterator(vector.data() + vector.size()))),
          2);
      auto outer = chunked.begin();
      auto inner = (*outer).begin();
      static_assert(std::same_as<const Iterator&, decltype(inner.base())>);
      std::same_as<const Iterator&> decltype(auto) underlying = std::as_const(inner).base();
      assert(std::addressof(underlying) == std::addressof(inner.base()));
      assert(base(underlying) == vector.data());
      assert(*underlying == 1);
      ++inner;
      assert(std::addressof(underlying) == std::addressof(inner.base()));
      assert(base(underlying) == vector.data() + 1);
      assert(*underlying == 2);
      ++outer;
      auto next_inner = (*outer).begin();
      assert(std::addressof(underlying) == std::addressof(std::as_const(next_inner).base()));
      assert(base(underlying) == vector.data() + 2);
      assert(*underlying == 3);
      ++next_inner;
      assert(base(underlying) == vector.data() + 3);
      assert(*underlying == 4);
      ++outer;
      assert(outer == chunked.end());
      assert(base(underlying) == vector.data() + vector.size());
    }

    // When range is single
    {
      std::ranges::chunk_view<std::ranges::subrange<Iterator, Sentinel>> chunked(
          std::ranges::subrange<Iterator, Sentinel>(
              Iterator(single_vector.data()), Sentinel(Iterator(single_vector.data() + single_vector.size()))),
          2);
      auto outer = chunked.begin();
      auto inner = (*outer).begin();
      static_assert(std::same_as<const Iterator&, decltype(inner.base())>);
      std::same_as<const Iterator&> decltype(auto) underlying = std::as_const(inner).base();
      assert(std::addressof(underlying) == std::addressof(inner.base()));
      assert(base(underlying) == single_vector.data());
      assert(*underlying == 1);
      ++inner;
      assert(inner == std::default_sentinel);
      assert(std::addressof(underlying) == std::addressof(inner.base()));
      assert(base(underlying) == single_vector.data() + single_vector.size());
    }
  }

  return true;
}

int main(int, char**) {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>();
  test<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>>();

  static_assert(test<cpp17_input_iterator<int*>>());
  static_assert(test<cpp20_input_iterator<int*>>());
  static_assert(test<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>());
  static_assert(test<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>>());

  return 0;
}
