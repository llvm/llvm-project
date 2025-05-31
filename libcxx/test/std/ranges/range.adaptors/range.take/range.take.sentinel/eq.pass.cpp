//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr bool operator==(const CI<Const>& y, const sentinel& x);
// template<bool OtherConst = !Const>
//   requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr bool operator==(const CI<OtherConst>& y, const sentinel& x);

#include <cassert>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_comparisons.h"
#include "test_iterators.h"
#include "test_range.h"

template <bool Const>
using MaybeConstIterator = cpp20_input_iterator<std::conditional_t<Const, const int*, int*>>;

template <bool Const>
class CrossConstComparableSentinel {
  using Base = std::conditional_t<Const, const int*, int*>;
  Base base_;

public:
  CrossConstComparableSentinel() = default;
  constexpr explicit CrossConstComparableSentinel(Base base) : base_(base) {}

  friend constexpr bool operator==(const MaybeConstIterator<Const>& it, const CrossConstComparableSentinel& se) {
    return base(it) == se.base_;
  }

  friend constexpr bool operator==(const MaybeConstIterator<!Const>& it, const CrossConstComparableSentinel& se) {
    return base(it) == se.base_;
  }
};

static_assert(std::sentinel_for<CrossConstComparableSentinel<true>, MaybeConstIterator<false>>);
static_assert(std::sentinel_for<CrossConstComparableSentinel<true>, MaybeConstIterator<true>>);
static_assert(std::sentinel_for<CrossConstComparableSentinel<false>, MaybeConstIterator<false>>);
static_assert(std::sentinel_for<CrossConstComparableSentinel<false>, MaybeConstIterator<true>>);

struct CrossConstComparableView : std::ranges::view_base {
  template <std::size_t N>
  constexpr explicit CrossConstComparableView(int (&arr)[N]) : b_(arr), e_(arr + N) {}

  constexpr MaybeConstIterator<false> begin() { return MaybeConstIterator<false>{b_}; }
  constexpr CrossConstComparableSentinel<false> end() { return CrossConstComparableSentinel<false>{e_}; }

  constexpr MaybeConstIterator<true> begin() const { return MaybeConstIterator<true>{b_}; }
  constexpr CrossConstComparableSentinel<true> end() const { return CrossConstComparableSentinel<true>{e_}; }

private:
  int* b_;
  int* e_;
};

static_assert(std::ranges::range<CrossConstComparableView>);
static_assert(std::ranges::range<const CrossConstComparableView>);

struct NonCrossConstComparableView : std::ranges::view_base {
  int* begin();
  sentinel_wrapper<int*> end();

  long* begin() const;
  sentinel_wrapper<long*> end() const;
};

static_assert(std::ranges::range<NonCrossConstComparableView>);
static_assert(std::ranges::range<const NonCrossConstComparableView>);

constexpr bool test() {
  int buffer[8]                      = {1, 2, 3, 4, 5, 6, 7, 8};
  using CrossConstComparableTakeView = std::ranges::take_view<CrossConstComparableView>;

  {   // Compare CI<Const> with sentinel<Const>
    { // Const == true
      AssertEqualityReturnBool<std::ranges::iterator_t<const CrossConstComparableTakeView>,
                               std::ranges::sentinel_t<const CrossConstComparableTakeView>>();
      const CrossConstComparableTakeView tv(CrossConstComparableView{buffer}, 4);
      assert(testEquality(std::ranges::next(tv.begin(), 4), tv.end(), true));
      assert(testEquality(tv.begin(), tv.end(), false));
    }

    { // Const == false
      AssertEqualityReturnBool<std::ranges::iterator_t<CrossConstComparableTakeView>,
                               std::ranges::sentinel_t<CrossConstComparableTakeView>>();
      CrossConstComparableTakeView tv(CrossConstComparableView{buffer}, 4);
      assert(testEquality(std::ranges::next(tv.begin(), 4), tv.end(), true));
      assert(testEquality(std::ranges::next(tv.begin(), 1), tv.end(), false));
    }
  }

  {   // Compare CI<Const> with sentinel<!Const>
    { // Const == true
      AssertEqualityReturnBool<std::ranges::iterator_t<const CrossConstComparableTakeView>,
                               std::ranges::sentinel_t<CrossConstComparableTakeView>>();
      CrossConstComparableTakeView tv(CrossConstComparableView{buffer}, 4);
      assert(testEquality(std::ranges::next(std::as_const(tv).begin(), 4), tv.end(), true));
      assert(testEquality(std::ranges::next(std::as_const(tv).begin(), 2), tv.end(), false));
    }

    { // Const == false
      AssertEqualityReturnBool<std::ranges::iterator_t<CrossConstComparableTakeView>,
                               std::ranges::sentinel_t<const CrossConstComparableTakeView>>();
      CrossConstComparableTakeView tv(CrossConstComparableView{buffer}, 4);
      assert(testEquality(std::ranges::next(tv.begin(), 4), std::as_const(tv).end(), true));
      assert(testEquality(std::ranges::next(tv.begin(), 3), std::as_const(tv).end(), false));
    }
  }

  { // Check invalid comparisons between CI<Const> and sentinel<!Const>
    using TakeView = std::ranges::take_view<NonCrossConstComparableView>;
    static_assert(
        !weakly_equality_comparable_with<std::ranges::iterator_t<const TakeView>, std::ranges::sentinel_t<TakeView>>);
    static_assert(
        !weakly_equality_comparable_with<std::ranges::iterator_t<TakeView>, std::ranges::sentinel_t<const TakeView>>);

    // Those should be valid
    static_assert(
        weakly_equality_comparable_with<std::ranges::iterator_t<TakeView>, std::ranges::sentinel_t<TakeView>>);
    static_assert(weakly_equality_comparable_with<std::ranges::iterator_t<const TakeView>,
                                                  std::ranges::sentinel_t<const TakeView>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
