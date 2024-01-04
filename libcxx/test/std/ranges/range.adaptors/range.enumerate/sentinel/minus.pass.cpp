//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

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
#include "../types.h"

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

constexpr bool test() { return true; }

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
