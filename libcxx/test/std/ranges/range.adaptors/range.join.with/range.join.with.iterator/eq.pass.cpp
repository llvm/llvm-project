//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// friend constexpr bool operator==(const iterator& x, const iterator& y)
//   requires ref-is-glvalue && forward_range<Base> &&
//            equality_comparable<InnerIter>;

#include <ranges>

#include <array>
#include <cassert>
#include <utility>

#include "../types.h"
#include "test_comparisons.h"

template <class I1, class I2 = I1>
concept CanEq = requires(const I1& i1, const I2& i2) {
  { i1 == i2 } -> std::same_as<bool>;
  { i2 == i1 } -> std::same_as<bool>;
  { i1 != i2 } -> std::same_as<bool>;
  { i2 != i1 } -> std::same_as<bool>;
};

constexpr bool test() {
  { // `V` and `Pattern` are not empty. Test return types too.
    using V       = std::array<std::array<int, 2>, 3>;
    using Pattern = std::array<long, 1>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    using Iter  = std::ranges::iterator_t<JWV>;
    using CIter = std::ranges::iterator_t<const JWV>;
    static_assert(!std::same_as<Iter, CIter>);
    static_assert(CanEq<Iter>);
    static_assert(CanEq<CIter>);
    static_assert(CanEq<Iter, CIter>);

    JWV jwv(V{{{9, 8}, {7, 6}, {5, 4}}}, Pattern{0L});

    Iter it1 = jwv.begin();
    assert(*it1 == 9);
    assert(testEquality(it1, it1, true));

    Iter it2 = std::ranges::prev(jwv.end());
    assert(*it2 == 4);
    assert(testEquality(it2, it2, true));
    assert(testEquality(it1, it2, false));

    CIter cit1 = std::as_const(jwv).begin();
    assert(*cit1 == 9);
    assert(testEquality(cit1, cit1, true));
    assert(testEquality(it1, cit1, true));
    assert(testEquality(it2, cit1, false));

    CIter cit2 = std::ranges::prev(std::as_const(jwv).end());
    assert(*cit2 == 4);
    assert(testEquality(cit2, cit2, true));
    assert(testEquality(cit1, cit2, false));
    assert(testEquality(it1, cit2, false));
    assert(testEquality(it2, cit2, true));

    // `it1.inner_it_` and `it2.inner_it_` are equal, but `it1.outer_it_` and `it2.outer_it_` are not.
    std::ranges::advance(it1, 2);
    assert(*it1 == 0);
    std::ranges::advance(it2, -2);
    assert(*it2 == 0);
    assert(testEquality(it1, it2, false));

    // `cit1.inner_it_` and `cit2.inner_it_` are equal, but `cit1.outer_it_` and `cit2.outer_it_` are not.
    std::ranges::advance(cit1, 2);
    assert(*cit1 == 0);
    assert(testEquality(it1, cit1, true));
    std::ranges::advance(cit2, -2);
    assert(*cit2 == 0);
    assert(testEquality(it2, cit2, true));
    assert(testEquality(cit1, cit2, false));

    // `it1.inner_it_` and `it2.inner_it_` are equal, `it1.outer_it_` and `it2.outer_it_` are equal too.
    // `it1.inner_it_index()` and `it2.inner_it_index()` are equal to 1.
    ++it1;
    assert(*it1 == 7);
    std::ranges::advance(it2, -2);
    assert(*it2 == 7);
    assert(testEquality(it1, it2, true));

    // `cit1.inner_it_` and `cit2.inner_it_` are equal, `cit1.outer_it_` and `cit2.outer_it_` are equal too.
    // `cit1.inner_it_index()` and `cit2.inner_it_index()` are equal to 1.
    ++cit1;
    assert(*cit1 == 7);
    assert(testEquality(it1, cit1, true));
    std::ranges::advance(cit2, -2);
    assert(*cit2 == 7);
    assert(testEquality(it2, cit2, true));
    assert(testEquality(cit1, cit2, true));

    // `it1.inner_it_` and `it2.inner_it_` are equal, `it1.outer_it_` and `it2.outer_it_` are equal too.
    // `it1.inner_it_index()` and `it2.inner_it_index()` are equal to 0.
    --it1;
    assert(*it1 == 0);
    --it2;
    assert(*it2 == 0);
    assert(testEquality(it1, it2, true));

    // `cit1.inner_it_` and `cit2.inner_it_` are equal, `cit1.outer_it_` and `cit2.outer_it_` are equal too.
    // `cit1.inner_it_index()` and `cit2.inner_it_index()` are equal to 0.
    --cit1;
    assert(*cit1 == 0);
    assert(testEquality(it1, cit1, true));
    --cit2;
    assert(*cit2 == 0);
    assert(testEquality(it2, cit2, true));
    assert(testEquality(cit2, cit2, true));
  }

  { // `InnerIter` models input iterator and equality comparable. `Pattern` is empty.
    using Inner   = BasicVectorView<int, ViewProperties{.common = false}, EqComparableInputIter>;
    using V       = std::vector<Inner>;
    using Pattern = std::ranges::empty_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    using Iter  = std::ranges::iterator_t<JWV>;
    using CIter = std::ranges::iterator_t<const JWV>;
    static_assert(!std::same_as<Iter, CIter>);
    static_assert(CanEq<Iter>);
    static_assert(CanEq<CIter>);
    static_assert(!CanEq<CIter, Iter>);

    JWV jwv(V{Inner{1, 2}, Inner{5, 6}, Inner{9, 0}}, Pattern{});

    {
      Iter it1 = jwv.begin();
      assert(*it1 == 1);
      Iter it2 = std::ranges::next(jwv.begin(), 2);
      assert(*it2 == 5);
      assert(testEquality(it1, it2, false));
      ++it1;
      ++it1;
      assert(testEquality(it1, it2, true));
      ++it1;
      assert(testEquality(it1, it2, false));
    }

    {
      CIter cit1 = std::as_const(jwv).begin();
      assert(*cit1 == 1);
      CIter cit2 = std::ranges::next(std::as_const(jwv).begin(), 2);
      assert(*cit2 == 5);
      assert(testEquality(cit1, cit2, false));
      ++cit1;
      ++cit1;
      assert(testEquality(cit1, cit2, true));
      ++cit1;
      assert(testEquality(cit1, cit2, false));
    }
  }

  { // `Pattern` is not empty. Some elements of `V` are.
    using Inner   = BasicVectorView<int, ViewProperties{.common = false}, EqComparableInputIter>;
    using V       = BasicVectorView<Inner, ViewProperties{}, forward_iterator>;
    using Pattern = BasicVectorView<int, ViewProperties{.common = false}, forward_iterator>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    using Iter  = std::ranges::iterator_t<JWV>;
    using CIter = std::ranges::iterator_t<const JWV>;
    static_assert(!std::same_as<Iter, CIter>);
    static_assert(CanEq<Iter>);
    static_assert(CanEq<CIter>);
    static_assert(!CanEq<CIter, Iter>);

    JWV jwv(V{Inner{1}, Inner{}, Inner{27}}, Pattern{0});

    {
      Iter it1 = jwv.begin();
      assert(*it1 == 1);
      ++it1;
      assert(*it1 == 0);
      Iter it2 = jwv.begin();
      assert(testEquality(it1, it2, false));
      ++it2;
      assert(testEquality(it1, it2, true));

      ++it2;
      assert(*it1 == *it2);
      assert(testEquality(it1, it2, false));

      std::ranges::advance(it1, 2);
      ++it2;
      assert(*it1 == *it2);
      assert(testEquality(it1, it2, true));
    }

    {
      CIter cit1 = std::as_const(jwv).begin();
      assert(*cit1 == 1);
      ++cit1;
      assert(*cit1 == 0);
      CIter cit2 = std::as_const(jwv).begin();
      assert(testEquality(cit1, cit2, false));
      ++cit2;
      assert(testEquality(cit1, cit2, true));

      ++cit2;
      assert(*cit1 == *cit2);
      assert(testEquality(cit1, cit2, false));

      std::ranges::advance(cit1, 2);
      ++cit2;
      assert(*cit1 == *cit2);
      assert(testEquality(cit1, cit2, true));
    }
  }

  { // `ref-is-glvalue` is false
    using Inner   = std::vector<int>;
    using V       = RvalueVector<Inner>;
    using Pattern = std::ranges::empty_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;
    using Iter    = std::ranges::iterator_t<JWV>;
    static_assert(!CanEq<Iter>);
  }

  { // `Base` does not model forward range
    using Inner   = std::vector<int>;
    using V       = BasicVectorView<Inner, ViewProperties{}, DefaultCtorInputIter>;
    using Pattern = std::ranges::empty_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;
    using Iter    = std::ranges::iterator_t<JWV>;
    static_assert(!CanEq<Iter>);
  }

  { // `InnerIter` does not model equality comparable
    using Inner   = BasicVectorView<int, ViewProperties{.common = false}, cpp20_input_iterator>;
    using V       = std::vector<Inner>;
    using Pattern = std::ranges::empty_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;
    using Iter    = std::ranges::iterator_t<JWV>;
    using CIter   = std::ranges::iterator_t<const JWV>;
    static_assert(!CanEq<Iter>);
    static_assert(!CanEq<CIter>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
