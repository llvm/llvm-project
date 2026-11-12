//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// template <class ...Ts> struct common_comparison_category
// template <class ...Ts> using common_comparison_category_t


#include <compare>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

const volatile void* volatile sink;

template <class Expect, class ...Args>
void test_cat() {
  using Cat = std::common_comparison_category<Args...>;
  using CatT = typename Cat::type;
  static_assert(std::is_same<CatT, std::common_comparison_category_t<Args...>>::value, "");
  static_assert(std::is_same<CatT, Expect>::value, "expected different category");
};


// [class.spaceship]p4: The 'common comparison type' U of a possibly-empty list
//   of 'n' types T0, T1, ..., TN, is defined as follows:
int main(int, char**) {
  using PO = std::partial_ordering;
  using WO = std::weak_ordering;
  using SO = std::strong_ordering;

  // [cmp.common]p2: The member typedef-name type denotes the common comparison
  /// type ([class.spaceship]) of Ts..., the expanded parameter pack, or void if
  // any element of Ts is not a comparison category type.
  {
    test_cat<void, void>();
    test_cat<void, int*>();
    test_cat<void, SO&>();
    test_cat<void, SO const>();
    test_cat<void, SO*>();
    test_cat<void, SO, void, SO>();
  }

  // [class.spaceship]p4.1: If at least one Ti is std::partial_ordering, U is
  // std::partial_ordering ([cmp.partialord]).
  {
    test_cat<PO, PO>();
    test_cat<PO, SO, PO, SO>();
    test_cat<PO, WO, PO, SO>();
  }

  // [class.spaceship]p4.2: Otherwise, if at least one Ti is std::weak_ordering,
  // U is std::weak_ordering
  {
    test_cat<WO, WO>();
    test_cat<WO, SO, WO, SO>();
  }

  // [class.spaceship]p4.3: Otherwise, U is std::strong_ordering.
  {
    test_cat<SO, SO>();
    test_cat<SO, SO, SO>();
  }

  // [cmp.common]p2, note 2: This is std::strong_ordering if the expansion is empty.
  // [class.spaceship]p4.3, note 2: In particular this is the result when n is 0.
  {
    test_cat<SO>(); // empty type list
  }

  return 0;
}
