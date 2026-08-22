//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr iterator(iterator<!Const> i)
//   requires Const && (convertible_to<iterator_t<First>, iterator_t<const First>> && ...
//                   && convertible_to<iterator_t<Vs>,    iterator_t<const Vs>>);

#include <array>
#include <cassert>
#include <ranges>
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

constexpr bool test() {
  std::array a{1, 2, 3};

  { // non-const to const conversion succeeds when all underlying iterators are convertible
    std::ranges::cartesian_product_view v(NonSimpleCommon{a}, NonSimpleCommon{a});
    auto it1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> it2 = it1;
    assert(it1 == it2);

    static_assert(!std::is_same_v<decltype(it1), decltype(it2)>);
    // Cannot go the other way (const -> non-const).
    static_assert(!std::constructible_from<decltype(it1), decltype(it2)>);
  }

  { // const-incompatible underlying iterator: neither direction is constructible
    std::ranges::cartesian_product_view v(ConstIterIncompatibleView{a});
    auto it1 = v.begin();
    auto it2 = std::as_const(v).begin();

    static_assert(!std::is_same_v<decltype(it1), decltype(it2)>);
    static_assert(!std::constructible_from<decltype(it1), decltype(it2)>);
    static_assert(!std::constructible_from<decltype(it2), decltype(it1)>);
  }

  { // 3-range conversion
    std::ranges::cartesian_product_view v(NonSimpleCommon{a}, NonSimpleCommon{a}, NonSimpleCommon{a});
    auto it1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> it2 = it1;
    assert(it1 == it2);
  }

  { // LWG3760: "cartesian_product_view::iterator's parent_ is never valid".
    // The converting constructor must propagate parent_. operator++ consults
    // parent_->bases_, which is not a constant expression if parent_ is null.
    std::ranges::cartesian_product_view v(NonSimpleCommon{a}, NonSimpleCommon{a});
    std::ranges::iterator_t<const decltype(v)> it = v.begin();

    ++it;
    assert(*it == std::tuple(1, 2));

    ++it;
    ++it; // wraps the inner range
    assert(*it == std::tuple(2, 1));

    assert(it - std::as_const(v).begin() == 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
