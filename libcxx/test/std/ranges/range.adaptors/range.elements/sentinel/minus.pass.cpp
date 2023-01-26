//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<bool OtherConst>
//   requires sized_sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr range_difference_t<maybe-const<OtherConst, V>>
//   operator-(const iterator<OtherConst>& x, const sentinel& y);
//
// template<bool OtherConst>
//   requires sized_sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr range_difference_t<maybe-const<OtherConst, V>>
//   operator-(const sentinel& x, const iterator<OtherConst>& y);

#include <cassert>
#include <concepts>
#include <functional>
#include <ranges>
#include <tuple>

#include "../types.h"

template <bool Const>
struct Iter {
  std::tuple<int>* it_;

  using value_type       = std::tuple<int>;
  using difference_type  = intptr_t;
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
struct SizedSent {
  std::tuple<int>* end_;

  constexpr bool operator==(const Iter<Const>& i) const { return i.it_ == end_; }

  friend constexpr auto operator-(const SizedSent& st, const Iter<Const>& it) { return st.end_ - it.it_; }

  friend constexpr auto operator-(const Iter<Const>& it, const SizedSent& st) { return it.it_ - st.end_; }
};

template <bool Const>
struct CrossSizedSent {
  std::tuple<int>* end_;

  template <bool C>
  constexpr bool operator==(const Iter<C>& i) const {
    return i.it_ == end_;
  }

  template <bool C>
  friend constexpr auto operator-(const CrossSizedSent& st, const Iter<C>& it) {
    return st.end_ - it.it_;
  }

  template <bool C>
  friend constexpr auto operator-(const Iter<C>& it, const CrossSizedSent& st) {
    return it.it_ - st.end_;
  }
};

template <template <bool> class It, template <bool> class St>
struct Range : TupleBufferView {
  using TupleBufferView::TupleBufferView;

  using iterator       = It<false>;
  using sentinel       = St<false>;
  using const_iterator = It<true>;
  using const_sentinel = St<true>;

  constexpr iterator begin() { return {buffer_}; }
  constexpr const_iterator begin() const { return {buffer_}; }
  constexpr sentinel end() { return sentinel{buffer_ + size_}; }
  constexpr const_sentinel end() const { return const_sentinel{buffer_ + size_}; }
};

template <class T, class U>
concept HasMinus = requires(const T t, const U u) { t - u; };

template <class BaseRange>
using ElementsView = std::ranges::elements_view<BaseRange, 0>;

template <class BaseRange>
using ElemIter = std::ranges::iterator_t<ElementsView<BaseRange>>;

template <class BaseRange>
using EleConstIter = std::ranges::iterator_t<const ElementsView<BaseRange>>;

template <class BaseRange>
using EleSent = std::ranges::sentinel_t<ElementsView<BaseRange>>;

template <class BaseRange>
using EleConstSent = std::ranges::sentinel_t<const ElementsView<BaseRange>>;

constexpr void testConstraints() {
  // base is not sized
  {
    using Base = Range<Iter, Sent>;
    static_assert(!HasMinus<EleSent<Base>, ElemIter<Base>>);
    static_assert(!HasMinus<ElemIter<Base>, EleSent<Base>>);

    static_assert(!HasMinus<EleSent<Base>, EleConstIter<Base>>);
    static_assert(!HasMinus<EleConstIter<Base>, EleSent<Base>>);

    static_assert(!HasMinus<EleConstSent<Base>, EleConstIter<Base>>);
    static_assert(!HasMinus<EleConstIter<Base>, EleConstSent<Base>>);

    static_assert(!HasMinus<EleConstSent<Base>, ElemIter<Base>>);
    static_assert(!HasMinus<ElemIter<Base>, EleConstSent<Base>>);
  }

  // base is sized but not cross const
  {
    using Base = Range<Iter, SizedSent>;
    static_assert(HasMinus<EleSent<Base>, ElemIter<Base>>);
    static_assert(HasMinus<ElemIter<Base>, EleSent<Base>>);

    static_assert(!HasMinus<EleSent<Base>, EleConstIter<Base>>);
    static_assert(!HasMinus<EleConstIter<Base>, EleSent<Base>>);

    static_assert(HasMinus<EleConstSent<Base>, EleConstIter<Base>>);
    static_assert(HasMinus<EleConstIter<Base>, EleConstSent<Base>>);

    static_assert(!HasMinus<EleConstSent<Base>, ElemIter<Base>>);
    static_assert(!HasMinus<ElemIter<Base>, EleConstSent<Base>>);
  }

  // base is cross const sized
  {
    using Base = Range<Iter, CrossSizedSent>;
    static_assert(HasMinus<EleSent<Base>, ElemIter<Base>>);
    static_assert(HasMinus<ElemIter<Base>, EleSent<Base>>);

    static_assert(HasMinus<EleSent<Base>, EleConstIter<Base>>);
    static_assert(HasMinus<EleConstIter<Base>, EleSent<Base>>);

    static_assert(HasMinus<EleConstSent<Base>, EleConstIter<Base>>);
    static_assert(HasMinus<EleConstIter<Base>, EleConstSent<Base>>);

    static_assert(HasMinus<EleConstSent<Base>, ElemIter<Base>>);
    static_assert(HasMinus<ElemIter<Base>, EleConstSent<Base>>);
  }
}

constexpr bool test() {
  std::tuple<int> buffer[] = {{1}, {2}, {3}, {4}, {5}};

  // base is sized but not cross const
  {
    using Base = Range<Iter, SizedSent>;
    Base base{buffer};
    auto ev         = base | std::views::elements<0>;
    auto iter       = ev.begin();
    auto const_iter = std::as_const(ev).begin();
    auto sent       = ev.end();
    auto const_sent = std::as_const(ev).end();

    assert(iter - sent == -5);
    assert(sent - iter == 5);
    assert(const_iter - const_sent == -5);
    assert(const_sent - const_iter == 5);
  }

  // base is cross const sized
  {
    using Base = Range<Iter, CrossSizedSent>;
    Base base{buffer};
    auto ev         = base | std::views::elements<0>;
    auto iter       = ev.begin();
    auto const_iter = std::as_const(ev).begin();
    auto sent       = ev.end();
    auto const_sent = std::as_const(ev).end();

    assert(iter - sent == -5);
    assert(sent - iter == 5);
    assert(iter - const_sent == -5);
    assert(const_sent - iter == 5);
    assert(const_iter - sent == -5);
    assert(sent - const_iter == 5);
    assert(const_iter - const_sent == -5);
    assert(const_sent - const_iter == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
