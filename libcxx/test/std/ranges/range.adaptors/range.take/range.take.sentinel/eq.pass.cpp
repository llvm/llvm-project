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

#include "test_iterators.h"

template <bool Const>
using MaybeConstIterator = cpp20_input_iterator<std::conditional_t<Const, const int*, int*>>;

template <bool Const>
class StrictSentinel {
  using Base = std::conditional_t<Const, const int*, int*>;
  Base base_;

public:
  StrictSentinel() = default;
  constexpr explicit StrictSentinel(Base base) : base_(base) {}

  friend constexpr bool operator==(const MaybeConstIterator<Const>& it, const StrictSentinel& se) {
    return base(it) == se.base_;
  }

  friend constexpr bool operator==(const MaybeConstIterator<!Const>& it, const StrictSentinel& se) {
    return base(it) == se.base_;
  }
};

static_assert(std::sentinel_for<StrictSentinel<true>, MaybeConstIterator<false>>);
static_assert(std::sentinel_for<StrictSentinel<true>, MaybeConstIterator<true>>);
static_assert(std::sentinel_for<StrictSentinel<false>, MaybeConstIterator<false>>);
static_assert(std::sentinel_for<StrictSentinel<false>, MaybeConstIterator<true>>);

struct CrossConstComparableView : std::ranges::view_base {
  template <std::size_t N>
  constexpr explicit CrossConstComparableView(int (&arr)[N]) : b_(arr), e_(arr + N) {}

  constexpr MaybeConstIterator<false> begin() { return MaybeConstIterator<false>{b_}; }
  constexpr StrictSentinel<false> end() { return StrictSentinel<false>{e_}; }

  constexpr MaybeConstIterator<true> begin() const { return MaybeConstIterator<true>{b_}; }
  constexpr StrictSentinel<true> end() const { return StrictSentinel<true>{e_}; }

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

template <class T, class U>
concept weakly_equality_comparable_with = requires(const T& t, const U& u) {
  t == u;
  t != u;
  u == t;
  u != t;
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {   // Compare CI<Const> with sentinel<Const>
    { // Const == true
      const std::ranges::take_view<CrossConstComparableView> tv(CrossConstComparableView{buffer}, 4);
      std::same_as<bool> decltype(auto) b1 = (tv.end() == std::ranges::next(tv.begin(), 4));
      assert(b1);
      std::same_as<bool> decltype(auto) b2 = (std::ranges::next(tv.begin(), 4) == tv.end());
      assert(b2);
      std::same_as<bool> decltype(auto) b3 = (tv.end() != tv.begin());
      assert(b3);
      std::same_as<bool> decltype(auto) b4 = (tv.begin() != tv.end());
      assert(b4);
      std::same_as<bool> decltype(auto) b5 = (std::ranges::next(tv.begin(), 1) == tv.end());
      assert(!b5);
      std::same_as<bool> decltype(auto) b6 = (std::ranges::next(tv.begin(), 4) != tv.end());
      assert(!b6);
    }

    { // Const == false
      std::ranges::take_view<CrossConstComparableView> tv(CrossConstComparableView{buffer}, 4);
      std::same_as<bool> decltype(auto) b1 = (tv.end() == std::ranges::next(tv.begin(), 4));
      assert(b1);
      std::same_as<bool> decltype(auto) b2 = (std::ranges::next(tv.begin(), 4) == tv.end());
      assert(b2);
      std::same_as<bool> decltype(auto) b3 = (tv.end() != tv.begin());
      assert(b3);
      std::same_as<bool> decltype(auto) b4 = (tv.begin() != tv.end());
      assert(b4);
      std::same_as<bool> decltype(auto) b5 = (std::ranges::next(tv.begin(), 2) == tv.end());
      assert(!b5);
      std::same_as<bool> decltype(auto) b6 = (std::ranges::next(tv.begin(), 4) != tv.end());
      assert(!b6);
    }
  }

  {   // Compare CI<Const> with sentinel<!Const>
    { // Const == true
      std::ranges::take_view<CrossConstComparableView> tv(CrossConstComparableView{buffer}, 4);
      std::same_as<bool> decltype(auto) b1 = (tv.end() == std::ranges::next(std::as_const(tv).begin(), 4));
      assert(b1);
      std::same_as<bool> decltype(auto) b2 = (std::ranges::next(std::as_const(tv).begin(), 4) == tv.end());
      assert(b2);
      std::same_as<bool> decltype(auto) b3 = (tv.end() != std::as_const(tv).begin());
      assert(b3);
      std::same_as<bool> decltype(auto) b4 = (std::as_const(tv).begin() != tv.end());
      assert(b4);
      std::same_as<bool> decltype(auto) b5 = (std::ranges::next(std::as_const(tv).begin(), 1) == tv.end());
      assert(!b5);
      std::same_as<bool> decltype(auto) b6 = (std::ranges::next(std::as_const(tv).begin(), 4) != tv.end());
      assert(!b6);
    }

    { // Const == false
      std::ranges::take_view<CrossConstComparableView> tv(CrossConstComparableView{buffer}, 4);
      std::same_as<bool> decltype(auto) b1 = (std::as_const(tv).end() == std::ranges::next(tv.begin(), 4));
      assert(b1);
      std::same_as<bool> decltype(auto) b2 = (std::ranges::next(tv.begin(), 4) == std::as_const(tv).end());
      assert(b2);
      std::same_as<bool> decltype(auto) b3 = (std::as_const(tv).end() != tv.begin());
      assert(b3);
      std::same_as<bool> decltype(auto) b4 = (tv.begin() != std::as_const(tv).end());
      assert(b4);
      std::same_as<bool> decltype(auto) b5 = (std::ranges::next(tv.begin(), 2) == std::as_const(tv).end());
      assert(!b5);
      std::same_as<bool> decltype(auto) b6 = (std::ranges::next(tv.begin(), 4) != std::as_const(tv).end());
      assert(!b6);
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
