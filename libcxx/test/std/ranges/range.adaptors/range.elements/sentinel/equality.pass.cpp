//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

//  template<bool OtherConst>
//    requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
//  friend constexpr bool operator==(const iterator<OtherConst>& x, const sentinel& y);

#include <array>
#include <cassert>
#include <ranges>

#include "../types.h"
#include "test_range.h"

template <bool Const>
struct Iter {
  std::tuple<int>* it_;

  using value_type       = std::tuple<int>;
  using difference_type  = std::intptr_t;
  using iterator_concept = std::input_iterator_tag;

  constexpr decltype(auto) operator*() const { return *it_; }
  constexpr Iter& operator++() {
    ++it_;
    return *this;
  }
  constexpr void operator++(int) { ++it_; }
};

template <bool Const>
struct Sent {
  std::tuple<int>* end_;

  constexpr bool operator==(const Iter<Const>& i) const { return i.it_ == end_; }
};

template <bool Const>
struct CrossComparableSent {
  std::tuple<int>* end_;

  template <bool C>
  constexpr bool operator==(const Iter<C>& i) const {
    return i.it_ == end_;
  }
};

template <template <bool> typename St>
struct Range : TupleBufferView {
  using TupleBufferView::TupleBufferView;
  constexpr Iter<false> begin() { return Iter<false>{buffer_}; }
  constexpr Iter<true> begin() const { return Iter<true>{buffer_}; }
  constexpr St<false> end() { return St<false>{buffer_ + size_}; }
  constexpr St<true> end() const { return St<true>{buffer_ + size_}; }
};

using R                = Range<Sent>;
using CrossComparableR = Range<CrossComparableSent>;

using std::ranges::elements_view;
using std::ranges::iterator_t;
using std::ranges::sentinel_t;

static_assert(weakly_equality_comparable_with<iterator_t<elements_view<R, 0>>, //
                                              sentinel_t<elements_view<R, 0>>>);

static_assert(!weakly_equality_comparable_with<iterator_t<const elements_view<R, 0>>, //
                                               sentinel_t<elements_view<R, 0>>>);

static_assert(!weakly_equality_comparable_with<iterator_t<elements_view<R, 0>>, //
                                               sentinel_t<const elements_view<R, 0>>>);

static_assert(weakly_equality_comparable_with<iterator_t<const elements_view<R, 0>>, //
                                              sentinel_t<const elements_view<R, 0>>>);

static_assert(weakly_equality_comparable_with<iterator_t<elements_view<R, 0>>, //
                                              sentinel_t<elements_view<R, 0>>>);

static_assert(weakly_equality_comparable_with<iterator_t<const elements_view<CrossComparableR, 0>>, //
                                              sentinel_t<elements_view<CrossComparableR, 0>>>);

static_assert(weakly_equality_comparable_with<iterator_t<elements_view<CrossComparableR, 0>>, //
                                              sentinel_t<const elements_view<CrossComparableR, 0>>>);

static_assert(weakly_equality_comparable_with<iterator_t<const elements_view<CrossComparableR, 0>>, //
                                              sentinel_t<const elements_view<CrossComparableR, 0>>>);

template <class R, bool ConstIter, bool ConstSent>
constexpr void testOne() {
  auto getBegin = [](auto&& rng) {
    if constexpr (ConstIter) {
      return std::as_const(rng).begin();
    } else {
      return rng.begin();
    }
  };

  auto getEnd = [](auto&& rng) {
    if constexpr (ConstSent) {
      return std::as_const(rng).end();
    } else {
      return rng.end();
    }
  };

  // iter == sentinel.base
  {
    std::tuple<int> buffer[] = {{1}};
    R v{buffer};
    std::ranges::elements_view<R, 0> ev(v);
    auto iter = getBegin(ev);
    auto st   = getEnd(ev);
    ++iter;
    assert(iter == st);
  }

  // iter != sentinel.base
  {
    std::tuple<int> buffer[] = {{1}};
    R v{buffer};
    std::ranges::elements_view<R, 0> ev(v);
    auto iter = getBegin(ev);
    auto st   = getEnd(ev);
    assert(iter != st);
  }

  // empty range
  {
    std::array<std::tuple<int>, 0> arr;
    R v{arr};
    std::ranges::elements_view<R, 0> ev(v);
    auto iter = getBegin(ev);
    auto sent = getEnd(ev);
    assert(iter == sent);
  }
}

constexpr bool test() {
  testOne<R, false, false>();
  testOne<R, true, true>();
  testOne<CrossComparableR, false, false>();
  testOne<CrossComparableR, true, true>();
  testOne<CrossComparableR, true, false>();
  testOne<CrossComparableR, false, true>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
