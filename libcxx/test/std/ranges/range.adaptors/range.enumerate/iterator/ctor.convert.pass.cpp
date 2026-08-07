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

struct TestIt {
  template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
  constexpr void operator()() const {
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
      // Assigning a non-const iterator to a const-iterator-typed variable invokes
      // the converting constructor.
      std::same_as<EnumerateConstIterator> decltype(auto) it = view.begin();
      std::same_as<const Iterator&> decltype(auto) itResult  = it.base();
      assert(base(base(itResult)) == std::to_address(base(array.begin())));

      auto [index, value] = *(++it);
      assert(index == 1);
      assert(value == 84);
    }
  }
};

constexpr bool test() {
  using Iterators =
      types::type_list< cpp17_input_iterator<int*>,
                        cpp20_input_iterator<int*>,
                        forward_iterator<int*>,
                        bidirectional_iterator<int*>,
                        random_access_iterator<int*>,
                        contiguous_iterator<int*>,
                        int* >;

  types::for_each(Iterators{}, TestIt());

  int buffer[3] = {1, 2, 3};
  {
    std::ranges::enumerate_view v(NonSimpleCommon{buffer});
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    assert(iter1 == iter2);

    static_assert(!std::same_as<decltype(iter1), decltype(iter2)>);

    // We cannot create a non-const iterator from a const iterator.
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
  }
  {
    // underlying non-const to const not convertible
    std::ranges::enumerate_view v(ConstIterIncompatibleView{buffer});
    auto iter1 = v.begin();
    auto iter2 = std::as_const(v).begin();

    static_assert(!std::same_as<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter2), decltype(iter1)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
