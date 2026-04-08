//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator(iterator<!Const> i)
//      requires Const && convertible_to<iterator_t<V>, iterator_t<const V>>;

#include <ranges>

#include <cassert>
#include <tuple>
#include <utility>

#include "../../range_adaptor_types.h"

using ConstIterIncompatibleView =
    BasicView<forward_iterator<int*>,
              forward_iterator<int*>,
              random_access_iterator<const int*>,
              random_access_iterator<const int*>>;
static_assert(!std::convertible_to<std::ranges::iterator_t<ConstIterIncompatibleView>,
                                   std::ranges::iterator_t<const ConstIterIncompatibleView>>);

template <std::size_t N>
constexpr void test() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::ranges::adjacent_view<NonSimpleCommon, N> v(NonSimpleCommon{buffer});
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    // We cannot create a non-const iterator from a const iterator.
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);

    assert(iter1 == iter2);

    auto tuple = *iter2;
    assert(std::get<0>(tuple) == buffer[0]);
    if constexpr (N >= 2)
      assert(std::get<1>(tuple) == buffer[1]);
    if constexpr (N >= 3)
      assert(std::get<2>(tuple) == buffer[2]);
    if constexpr (N >= 4)
      assert(std::get<3>(tuple) == buffer[3]);
    if constexpr (N >= 5)
      assert(std::get<4>(tuple) == buffer[4]);
  }

  {
    // underlying non-const to const not convertible
    std::ranges::adjacent_view<ConstIterIncompatibleView, N> v(ConstIterIncompatibleView{buffer});
    auto iter1 = v.begin();
    auto iter2 = std::as_const(v).begin();

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);

    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter2), decltype(iter1)>);
  }
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<5>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
