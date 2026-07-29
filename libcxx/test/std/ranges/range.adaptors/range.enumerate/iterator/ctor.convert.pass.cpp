//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// class enumerate_view::iterator

// constexpr iterator(iterator<!Const> i)
//   requires Const && convertible_to<iterator_t<V>, iterator_t<Base>>;

#include <array>
#include <cassert>
#include <concepts>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_iterators.h"

#include "../../range_adaptor_types.h"
#include "../types.h"

using ConstIterIncompatibleView =
    BasicView<forward_iterator<int*>,
              forward_iterator<int*>,
              random_access_iterator<const int*>,
              random_access_iterator<const int*>>;
static_assert(!std::convertible_to<std::ranges::iterator_t<ConstIterIncompatibleView>,
                                   std::ranges::iterator_t<const ConstIterIncompatibleView>>);

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View                   = MinimalView<Iterator, Sentinel>;
  using EnumerateView          = std::ranges::enumerate_view<View>;
  using EnumerateIterator      = std::ranges::iterator_t<EnumerateView>;
  using EnumerateConstIterator = std::ranges::iterator_t<const EnumerateView>;

  auto make_enumerate_view = [](auto begin, auto end) {
    View view{Iterator(std::to_address(base(begin))), Sentinel(Iterator(std::to_address(base(end))))};

    return EnumerateView(std::move(view));
  };

  static_assert(std::convertible_to<EnumerateIterator, EnumerateConstIterator>);

  std::array array{0, 84, 2, 3, 4};
  auto view = make_enumerate_view(array.begin(), array.end());
  {
    std::same_as<EnumerateIterator> decltype(auto) it     = view.begin();
    std::same_as<const Iterator&> decltype(auto) itResult = it.base();
    assert(base(base(itResult)) == std::to_address(base(array.begin())));

    auto [index, value] = *(++it);
    assert(index == 1);
    assert(value == 84);
  }
  {
    std::same_as<EnumerateConstIterator> decltype(auto) it = view.begin();
    std::same_as<const Iterator&> decltype(auto) itResult  = it.base();
    assert(base(base(itResult)) == std::to_address(base(array.begin())));

    auto [index, value] = *(++it);
    assert(index == 1);
    assert(value == 84);
  }
  {
    // underlying non-const to const not convertible
    int buffer[3] = {1, 2, 3};
    std::ranges::enumerate_view v(ConstIterIncompatibleView{buffer});
    auto iter1 = v.begin();
    auto iter2 = std::as_const(v).begin();

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter2), decltype(iter1)>);
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
