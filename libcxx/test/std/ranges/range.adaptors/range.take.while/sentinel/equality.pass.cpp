//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

//  friend constexpr bool operator==(const iterator_t<Base>& x, const sentinel& y);
//
//  template<bool OtherConst = !Const>
//    requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
//  friend constexpr bool operator==(const iterator_t<maybe-const<OtherConst, V>>& x,
//                                   const sentinel& y);

#include <array>
#include <cassert>
#include <ranges>

#include "../types.h"

template <bool Const>
struct Iter {
  int* it_;

  using value_type       = int;
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
  int* end_;

  constexpr bool operator==(const Iter<Const>& i) const { return i.it_ == end_; }
};

template <bool Const>
struct CrossComparableSent {
  int* end_;

  template <bool C>
  constexpr bool operator==(const Iter<C>& i) const {
    return i.it_ == end_;
  }
};

template <template <bool> typename St>
struct Range : IntBufferViewBase {
  using IntBufferViewBase::IntBufferViewBase;
  constexpr Iter<false> begin() { return Iter<false>{buffer_}; }
  constexpr Iter<true> begin() const { return Iter<true>{buffer_}; }
  constexpr St<false> end() { return St<false>{buffer_ + size_}; }
  constexpr St<true> end() const { return St<true>{buffer_ + size_}; }
};

using R                = Range<Sent>;
using CrossComparableR = Range<CrossComparableSent>;

struct LessThan3 {
  constexpr bool operator()(int i) const { return i < 3; }
};

using std::ranges::iterator_t;
using std::ranges::sentinel_t;
using std::ranges::take_while_view;

static_assert(weakly_equality_comparable_with<iterator_t<take_while_view<R, LessThan3>>, //
                                              sentinel_t<take_while_view<R, LessThan3>>>);

static_assert(!weakly_equality_comparable_with<iterator_t<const take_while_view<R, LessThan3>>, //
                                               sentinel_t<take_while_view<R, LessThan3>>>);

static_assert(!weakly_equality_comparable_with<iterator_t<take_while_view<R, LessThan3>>, //
                                               sentinel_t<const take_while_view<R, LessThan3>>>);

static_assert(weakly_equality_comparable_with<iterator_t<const take_while_view<R, LessThan3>>, //
                                              sentinel_t<const take_while_view<R, LessThan3>>>);

static_assert(weakly_equality_comparable_with<iterator_t<take_while_view<CrossComparableR, LessThan3>>, //
                                              sentinel_t<take_while_view<CrossComparableR, LessThan3>>>);

static_assert(weakly_equality_comparable_with<iterator_t<const take_while_view<CrossComparableR, LessThan3>>, //
                                              sentinel_t<take_while_view<CrossComparableR, LessThan3>>>);

static_assert(weakly_equality_comparable_with<iterator_t<take_while_view<CrossComparableR, LessThan3>>, //
                                              sentinel_t<const take_while_view<CrossComparableR, LessThan3>>>);

static_assert(weakly_equality_comparable_with<iterator_t<const take_while_view<CrossComparableR, LessThan3>>, //
                                              sentinel_t<const take_while_view<CrossComparableR, LessThan3>>>);

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
    int buffer[] = {1};
    R v{buffer};
    std::ranges::take_while_view twv(v, LessThan3{});
    auto iter = getBegin(twv);
    auto st   = getEnd(twv);
    ++iter;
    assert(iter == st);
  }

  // iter != sentinel.base && pred(*iter)
  {
    int buffer[] = {1, 3, 4};
    R v{buffer};
    std::ranges::take_while_view twv(v, LessThan3{});
    auto iter = getBegin(twv);
    auto st   = getEnd(twv);
    assert(iter != st);
    ++iter;
    assert(iter == st);
  }

  // iter != sentinel.base && !pred(*iter)
  {
    int buffer[] = {1, 2, 3, 4, 3, 2, 1};
    R v{buffer};
    std::ranges::take_while_view twv(v, LessThan3{});
    auto iter = getBegin(twv);
    auto sent = getEnd(twv);
    assert(iter != sent);
  }

  // empty range
  {
    std::array<int, 0> arr;
    R v{arr};
    std::ranges::take_while_view twv(v, LessThan3{});
    auto iter = getBegin(twv);
    auto sent = getEnd(twv);
    assert(iter == sent);
  }
}

constexpr bool test() {
  testOne<R, false, false>();
  testOne<R, true, true>();
  testOne<CrossComparableR, false, false>();
  testOne<CrossComparableR, true, true>();

  // LWG 3449 `take_view` and `take_while_view`'s `sentinel<false>` not comparable with their const iterator
  testOne<CrossComparableR, true, false>();
  testOne<CrossComparableR, false, true>();

  // test std::invoke is used
  {
    struct Data {
      bool b;
    };

    Data buffer[] = {{true}, {true}, {false}};
    std::ranges::take_while_view twv(buffer, &Data::b);
    auto it = twv.begin();
    auto st = twv.end();
    assert(it != st);

    ++it;
    assert(it != st);

    ++it;
    assert(it == st);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
