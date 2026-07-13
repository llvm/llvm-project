//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  constexpr iterator(iterator<!Const> i)
//    requires Const && convertible_to<inner-iterator<false>, inner-iterator<Const>>;

#include <ranges>

#include <cassert>
#include <tuple>
#include <utility>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

using ConstIterIncompatibleView =
    BasicView<forward_iterator<int*>,
              forward_iterator<int*>,
              random_access_iterator<const int*>,
              random_access_iterator<const int*>>;
static_assert(!std::convertible_to<std::ranges::iterator_t<ConstIterIncompatibleView>,
                                   std::ranges::iterator_t<const ConstIterIncompatibleView>>);

template <std::size_t N, class Fn, class Validator>
constexpr void test() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};
  Validator validator{};
  {
    std::ranges::adjacent_transform_view<NonSimpleCommon, Fn, N> v(NonSimpleCommon{buffer}, Fn{});
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    // We cannot create a non-const iterator from a const iterator.
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);

    assert(iter1 == iter2);

    validator(buffer, *iter1, 0);
    validator(buffer, *iter2, 0);
  }

  {
    // underlying non-const to const not convertible
    std::ranges::adjacent_transform_view<ConstIterIncompatibleView, Tie, N> v(ConstIterIncompatibleView{buffer}, Tie{});
    auto iter1 = v.begin();
    auto iter2 = std::as_const(v).begin();

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);

    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter2), decltype(iter1)>);
  }
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple, ValidateTupleFromIndex<N>>();
  test<N, Tie, ValidateTieFromIndex<N>>();
  test<N, GetFirst, ValidateGetFirstFromIndex<N>>();
  test<N, Multiply, ValidateMultiplyFromIndex<N>>();
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
