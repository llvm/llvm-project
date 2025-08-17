//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// template<bool OtherConst>
//   requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr bool operator==(const iterator<OtherConst>& x, const sentinel& y);

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

#include "../types.h"
#include "test_comparisons.h"
#include "test_iterators.h"

struct NonCrossConstComparableView : std::ranges::view_base {
  using NonConstRange = std::vector<int>;
  NonConstRange* begin();
  sentinel_wrapper<NonConstRange*> end();

  using ConstRange = BasicVectorView<int, ViewProperties{}, forward_iterator>;
  ConstRange* begin() const;
  sentinel_wrapper<ConstRange*> end() const;
};

static_assert(std::ranges::range<NonCrossConstComparableView>);
static_assert(std::ranges::range<const NonCrossConstComparableView>);

constexpr bool test() {
  using Inner   = BasicVectorView<int, ViewProperties{.common = false}, cpp20_input_iterator>;
  using V       = std::vector<Inner>;
  using Pattern = std::ranges::single_view<int>;
  using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, Pattern>;
  static_assert(!std::ranges::common_range<JWV>);

  using Iter  = std::ranges::iterator_t<JWV>;
  using CIter = std::ranges::iterator_t<const JWV>;
  static_assert(!std::same_as<Iter, CIter>);

  using Sent  = std::ranges::sentinel_t<JWV>;
  using CSent = std::ranges::sentinel_t<const JWV>;
  static_assert(!std::same_as<Sent, CSent>);

  {   // Compare iterator<Const> with sentinel<Const>
    { // Const == true
      AssertEqualityReturnBool<CIter, CSent>();
      const JWV jwv(V{Inner{1, 2}, Inner{4}}, 3);
      assert(testEquality(std::ranges::next(jwv.begin(), 4), jwv.end(), true));
      assert(testEquality(jwv.begin(), jwv.end(), false));
    }

    { // Const == false
      AssertEqualityReturnBool<Iter, Sent>();
      JWV jwv(V{Inner{5}, Inner{7, 8}}, 6);
      assert(testEquality(std::ranges::next(jwv.begin(), 4), jwv.end(), true));
      assert(testEquality(std::ranges::next(jwv.begin(), 2), jwv.end(), false));
    }
  }

  {   // Compare iterator<Const> with sentinel<!Const>
    { // Const == true
      AssertEqualityReturnBool<CIter, Sent>();
      JWV jwv(V{Inner{9, 10}, Inner{12}}, 11);
      assert(testEquality(std::ranges::next(std::as_const(jwv).begin(), 4), jwv.end(), true));
      assert(testEquality(std::ranges::next(std::as_const(jwv).begin(), 2), jwv.end(), false));
    }

    { // Const == false
      AssertEqualityReturnBool<Iter, CSent>();
      JWV jwv(V{Inner{13}, Inner{15, 16}}, 14);
      assert(testEquality(std::ranges::next(jwv.begin(), 4), std::as_const(jwv).end(), true));
      assert(testEquality(std::ranges::next(jwv.begin(), 3), std::as_const(jwv).end(), false));
    }
  }

  { // Check invalid comparisons between iterator<Const> and sentinel<!Const>
    using JWV2 = std::ranges::join_with_view<NonCrossConstComparableView, Pattern>;
    static_assert(!std::ranges::common_range<JWV2>);

    static_assert(!weakly_equality_comparable_with<std::ranges::iterator_t<const JWV2>, std::ranges::sentinel_t<JWV2>>);
    static_assert(!weakly_equality_comparable_with<std::ranges::iterator_t<JWV2>, std::ranges::sentinel_t<const JWV2>>);

    // Those should be valid
    static_assert(weakly_equality_comparable_with<std::ranges::iterator_t<JWV2>, std::ranges::sentinel_t<JWV2>>);
    static_assert(
        weakly_equality_comparable_with<std::ranges::iterator_t<const JWV2>, std::ranges::sentinel_t<const JWV2>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
