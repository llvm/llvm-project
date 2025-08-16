//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// class enumerate_view::sentinel

// template<bool OtherConst>
//   requires sized_sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr range_difference_t<maybe-const<OtherConst, V>>
//   operator-(const iterator<OtherConst>& x, const sentinel& y);

// template<bool OtherConst>
//   requires sized_sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr range_difference_t<maybe-const<OtherConst, V>>
//   operator-(const sentinel& x, const iterator<OtherConst>& y);

#include <array>
#include <cassert>
#include <ranges>

#include "test_iterators.h"

#include "../test_concepts.h"
// #include "../types_iterators.h"
#include "../types.h"

template <bool Const>
struct Iter {
  int* it_;

  using value_type       = int;
  using difference_type  = std::ptrdiff_t;
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
struct SizedSent {
  int* end_;

  constexpr bool operator==(const Iter<Const>& i) const { return i.it_ == end_; }

  friend constexpr auto operator-(const SizedSent& st, const Iter<Const>& it) { return st.end_ - it.it_; }

  friend constexpr auto operator-(const Iter<Const>& it, const SizedSent& st) { return it.it_ - st.end_; }
};

template <bool Const>
struct CrossSizedSent {
  int* end_;

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
struct BufferView : std::ranges::view_base {
  template <std::size_t N>
  constexpr BufferView(int (&b)[N]) : buffer_(b), size_(N) {}

  template <std::size_t N>
  constexpr BufferView(std::array<int, N>& arr) : buffer_(arr.data()), size_(N) {}

  using iterator       = It<false>;
  using sentinel       = St<false>;
  using const_iterator = It<true>;
  using const_sentinel = St<true>;

  constexpr iterator begin() { return {buffer_}; }
  constexpr const_iterator begin() const { return {buffer_}; }
  constexpr sentinel end() { return sentinel{buffer_ + size_}; }
  constexpr const_sentinel end() const { return const_sentinel{buffer_ + size_}; }

  int* buffer_;
  std::size_t size_;
};

template <template <bool> class It, template <bool> class St>
struct SizedBufferView : BufferView<It, St> {
  using BufferView<It, St>::BufferView;

  using typename BufferView<It, St>::iterator;
  using typename BufferView<It, St>::sentinel;
  using typename BufferView<It, St>::const_iterator;
  using typename BufferView<It, St>::const_sentinel;

  using BufferView<It, St>::begin;
  using BufferView<It, St>::end;

  constexpr std::size_t size() { return BufferView<It, St>::size_; }
};

template <class T, class U>
concept HasMinus = requires(const T t, const U u) { t - u; };

template <class BaseView>
using EnumerateView = std::ranges::enumerate_view<BaseView>;

template <class BaseView>
using EnumerateIter = std::ranges::iterator_t<EnumerateView<BaseView>>;

template <class BaseView>
using EnumerateConstIter = std::ranges::iterator_t<const EnumerateView<BaseView>>;

template <class BaseView>
using EnumerateSentinel = std::ranges::sentinel_t<EnumerateView<BaseView>>;

template <class BaseView>
using EnumerateConstSentinel = std::ranges::sentinel_t<const EnumerateView<BaseView>>;

constexpr void testConstraints() {
  // Base is not sized
  {
    using Base = BufferView<Iter, Sent>;

    static_assert(!HasMemberSize<Base>);
    static_assert(!std::ranges::sized_range<Base>);

    static_assert(!HasMinus<EnumerateIter<Base>, EnumerateSentinel<Base>>);
    static_assert(!HasMinus<EnumerateIter<Base>, EnumerateConstSentinel<Base>>);

    static_assert(!HasMinus<EnumerateConstIter<Base>, EnumerateSentinel<Base>>);
    static_assert(!HasMinus<EnumerateConstIter<Base>, EnumerateConstSentinel<Base>>);

    static_assert(!HasMinus<EnumerateSentinel<Base>, EnumerateIter<Base>>);
    static_assert(!HasMinus<EnumerateSentinel<Base>, EnumerateConstIter<Base>>);

    static_assert(!HasMinus<EnumerateConstSentinel<Base>, EnumerateIter<Base>>);
    static_assert(!HasMinus<EnumerateConstSentinel<Base>, EnumerateConstIter<Base>>);
  }

  // Base is sized but not cross const
  {
    using Base = SizedBufferView<Iter, Sent>;

    static_assert(HasMemberSize<Base>);
    static_assert(std::ranges::sized_range<Base>);

    // static_assert(HasMinus<EnumerateIter<Base>, EnumerateSentinel<Base>>);
    static_assert(!HasMinus<EnumerateIter<Base>, EnumerateConstSentinel<Base>>);

    static_assert(!HasMinus<EnumerateConstIter<Base>, EnumerateSentinel<Base>>);
    // static_assert(HasMinus<EnumerateConstIter<Base>, EnumerateConstSentinel<Base>>);

    // static_assert(HasMinus<EnumerateSentinel<Base>, EnumerateIter<Base>>);
    static_assert(!HasMinus<EnumerateSentinel<Base>, EnumerateConstIter<Base>>);

    static_assert(!HasMinus<EnumerateConstSentinel<Base>, EnumerateIter<Base>>);
    // static_assert(HasMinus<EnumerateConstSentinel<Base>, EnumerateConstIter<Base>>);
  }

  // Base is cross const sized
  {
    using Base = BufferView<Iter, CrossSizedSent>;

    static_assert(!HasMemberSize<Base>);
    static_assert(!std::ranges::sized_range<Base>);

    static_assert(HasMinus<EnumerateIter<Base>, EnumerateSentinel<Base>>);
    static_assert(HasMinus<EnumerateIter<Base>, EnumerateConstSentinel<Base>>);

    static_assert(HasMinus<EnumerateConstIter<Base>, EnumerateSentinel<Base>>);
    static_assert(HasMinus<EnumerateConstIter<Base>, EnumerateConstSentinel<Base>>);

    static_assert(HasMinus<EnumerateSentinel<Base>, EnumerateIter<Base>>);
    static_assert(HasMinus<EnumerateSentinel<Base>, EnumerateConstIter<Base>>);

    static_assert(HasMinus<EnumerateConstSentinel<Base>, EnumerateIter<Base>>);
    static_assert(HasMinus<EnumerateConstSentinel<Base>, EnumerateConstIter<Base>>);
  }
}

constexpr bool test() {
  int buffer[] = {1, 2, 3, 4, 5};

  // Base is sized but not cross const
  {
    using Base = SizedBufferView<Iter, SizedSent>;

    static_assert(HasMemberSize<Base>);
    static_assert(std::ranges::sized_range<Base>);

    Base base{buffer};
    auto ev         = base | std::views::enumerate;
    auto iter       = ev.begin();
    auto const_iter = std::as_const(ev).begin();
    auto sent       = ev.end();
    auto const_sent = std::as_const(ev).end();

    // Asssert difference
    assert(iter - sent == -5);
    assert(sent - iter == 5);
    assert(const_iter - const_sent == -5);
    assert(const_sent - const_iter == 5);
  }

  // Base is cross const sized
  {
    using Base = BufferView<Iter, CrossSizedSent>;

    static_assert(!HasMemberSize<Base>);
    static_assert(!std::ranges::sized_range<Base>);

    Base base{buffer};
    auto ev         = base | std::views::enumerate;
    auto iter       = ev.begin();
    auto const_iter = std::as_const(ev).begin();
    auto sent       = ev.end();
    auto const_sent = std::as_const(ev).end();

    // Assert difference
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
