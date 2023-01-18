//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator(iterator<!Const> i)
//   requires Const && convertible_to<iterator_t<V>, iterator_t<Base>>;

#include <cassert>
#include <ranges>
#include <tuple>

#include "../types.h"

template <bool Const>
struct ConvertibleIter : IterBase<ConvertibleIter<Const>> {
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = std::tuple<int>;
  using difference_type   = intptr_t;

  bool movedFromOtherConst = false;
  int i                    = 0;

  constexpr ConvertibleIter() = default;
  constexpr ConvertibleIter(int ii) : i(ii) {}
  template <bool otherConst>
    requires(Const != otherConst)
  constexpr ConvertibleIter(ConvertibleIter<otherConst> it) : movedFromOtherConst(true), i(it.i) {}
};

template <class Iter, class ConstIter>
struct BasicView : std::ranges::view_base {
  Iter begin();
  Iter end();

  ConstIter begin() const;
  ConstIter end() const;
};

template <class View>
using ElemIter = std::ranges::iterator_t<std::ranges::elements_view<View, 0>>;

template <class View>
using ConstElemIter = std::ranges::iterator_t<const std::ranges::elements_view<View, 0>>;

using ConvertibleView    = BasicView<ConvertibleIter<false>, ConvertibleIter<true>>;
using NonConvertibleView = BasicView<forward_iterator<std::tuple<int>*>, bidirectional_iterator<std::tuple<int>*>>;

static_assert(std::is_constructible_v<ConstElemIter<ConvertibleView>, ElemIter<ConvertibleView>>);
static_assert(!std::is_constructible_v<ElemIter<ConvertibleView>, ConstElemIter<ConvertibleView>>);
static_assert(!std::is_constructible_v<ConstElemIter<NonConvertibleView>, ElemIter<NonConvertibleView>>);
static_assert(!std::is_constructible_v<ElemIter<NonConvertibleView>, ConstElemIter<NonConvertibleView>>);

constexpr bool test() {
  ElemIter<ConvertibleView> iter{ConvertibleIter<false>{5}};
  ConstElemIter<ConvertibleView> constIter = iter; // implicit
  assert(constIter.base().movedFromOtherConst);
  assert(constIter.base().i == 5);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
