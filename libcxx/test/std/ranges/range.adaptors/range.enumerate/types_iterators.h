//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_TYPES_ITERATORS_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_TYPES_ITERATORS_H

#include "types.h"

// Iterators & Sentinels

template <bool Const>
struct Iterator {
  using value_type       = int;
  using difference_type  = std::ptrdiff_t;
  using iterator_concept = std::input_iterator_tag;

  constexpr decltype(auto) operator*() const { return *it_; }
  constexpr Iterator& operator++() {
    ++it_;

    return *this;
  }
  constexpr void operator++(int) { ++it_; }

  std::tuple<std::ptrdiff_t, int>* it_;
};

template <bool Const>
struct Sentinel {
  constexpr bool operator==(const Iterator<Const>& i) const { return i.it_ == end_; }

  std::tuple<std::ptrdiff_t, int>* end_;
};

template <bool Const>
struct CrossComparableSentinel {
  template <bool C>
  constexpr bool operator==(const Iterator<C>& i) const {
    return i.it_ == end_;
  }

  std::tuple<std::ptrdiff_t, int>* end_;
};

template <bool Const>
struct SizedSentinel {
  constexpr bool operator==(const Iterator<Const>& i) const { return i.it_ == end_; }

  friend constexpr auto operator-(const SizedSentinel& st, const Iterator<Const>& it) { return st.end_ - it.it_; }

  friend constexpr auto operator-(const Iterator<Const>& it, const SizedSentinel& st) { return it.it_ - st.end_; }

  int* end_;
};

template <bool Const>
struct CrossSizedSentinel {
  template <bool C>
  constexpr bool operator==(const Iterator<C>& i) const {
    return i.it_ == end_;
  }

  template <bool C>
  friend constexpr auto operator-(const CrossSizedSentinel& st, const Iterator<C>& it) {
    return st.end_ - it.it_;
  }

  template <bool C>
  friend constexpr auto operator-(const Iterator<C>& it, const CrossSizedSentinel& st) {
    return it.it_ - st.end_;
  }

  int* end_;
};

// Views

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

  using difference_type = int;
  using iterator_type   = std::tuple<std::ptrdiff_t, int>;

  constexpr iterator begin() { return iterator_type{pos_, buffer_}; }
  constexpr const_iterator begin() const { return {iterator_type{pos_, buffer_}}; }
  constexpr sentinel end() { return sentinel{buffer_ + size_}; }
  constexpr const_sentinel end() const { return const_sentinel{pos_, buffer_ + size_}; }

  std::ptrdiff_t pos_;
  int* buffer_;
  std::size_t size_;
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_TYPES_ITERATORS_H
