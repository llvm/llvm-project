//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// When the debug mode is enabled, we don't unwrap iterators in std::copy
// so we don't get this optimization.
// UNSUPPORTED: libcpp-has-debug-mode

// <algorithm>

// This test checks that std::copy forwards to memmove when appropriate.

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ranges>
#include <type_traits>

struct S {
  int i;
  constexpr S(int i_) : i(i_) {}
  S(const S&) = default;
  S(S&&) = delete;
  constexpr S& operator=(const S&) = default;
  S& operator=(S&&) = delete;
  constexpr bool operator==(const S&) const = default;
};

static_assert(std::is_trivially_copyable_v<S>);

template <class T>
struct NotIncrementableIt {
  T* i;
  using iterator_category = std::contiguous_iterator_tag;
  using iterator_concept = std::contiguous_iterator_tag;
  using value_type = T;
  using difference_type = ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  constexpr NotIncrementableIt() = default;
  constexpr NotIncrementableIt(T* i_) : i(i_) {}

  friend constexpr bool operator==(const NotIncrementableIt& lhs, const NotIncrementableIt& rhs) {
    return lhs.i == rhs.i;
  }

  constexpr T& operator*() { return *i; }
  constexpr T& operator*() const { return *i; }
  constexpr T* operator->() { return i; }
  constexpr T* operator->() const { return i; }

  constexpr NotIncrementableIt& operator++() {
    assert(false);
    return *this;
  }

  constexpr NotIncrementableIt& operator++(int) {
    assert(false);
    return *this;
  }

  constexpr NotIncrementableIt& operator--() {
    assert(false);
    return *this;
  }

  friend constexpr NotIncrementableIt operator+(const NotIncrementableIt& it, difference_type size) { return it.i + size; }
  friend constexpr difference_type operator-(const NotIncrementableIt& x, const NotIncrementableIt& y) { return x.i - y.i; }
  friend constexpr NotIncrementableIt operator-(const NotIncrementableIt& x, difference_type size) { return NotIncrementableIt(x.i - size); }
};

static_assert(std::__is_cpp17_contiguous_iterator<NotIncrementableIt<S>>::value);

template <size_t N, class Iter, std::enable_if_t<N == 0>* = nullptr>
constexpr auto wrap_n_times(Iter i) {
  return i;
}

template <size_t N, class Iter, std::enable_if_t<N != 0>* = nullptr>
constexpr auto wrap_n_times(Iter i) {
  return std::make_reverse_iterator(wrap_n_times<N - 1>(i));
}

static_assert(std::is_same_v<decltype(wrap_n_times<2>(std::declval<int*>())),
                             std::reverse_iterator<std::reverse_iterator<int*>>>);

template <size_t InCount, size_t OutCount, class Iter>
constexpr void test_normal() {
  {
    S a[] = {1, 2, 3, 4};
    S b[] = {0, 0, 0, 0};
    std::copy(wrap_n_times<InCount>(Iter(a)), wrap_n_times<InCount>(Iter(a + 4)), wrap_n_times<OutCount>(Iter(b)));
    assert(std::equal(a, a + 4, b));
  }
  {
    S a[] = {1, 2, 3, 4};
    S b[] = {0, 0, 0, 0};
    std::ranges::copy(wrap_n_times<InCount>(Iter(a)),
                      wrap_n_times<InCount>(Iter(a + 4)),
                      wrap_n_times<OutCount>(Iter(b)));
    assert(std::equal(a, a + 4, b));
  }
  {
    S a[] = {1, 2, 3, 4};
    S b[] = {0, 0, 0, 0};
    auto range = std::ranges::subrange(wrap_n_times<InCount>(Iter(a)), wrap_n_times<InCount>(Iter(a + 4)));
    std::ranges::copy(range, Iter(b));
    assert(std::equal(a, a + 4, b));
  }
}

template <size_t InCount, size_t OutCount, class Iter>
constexpr void test_reverse() {
  {
    S a[] = {1, 2, 3, 4};
    S b[] = {0, 0, 0, 0};
    std::copy(std::make_reverse_iterator(wrap_n_times<InCount>(Iter(a + 4))),
              std::make_reverse_iterator(wrap_n_times<InCount>(Iter(a))),
              std::make_reverse_iterator(wrap_n_times<OutCount>(Iter(b + 4))));
    assert(std::equal(a, a + 4, b));
  }
  {
    S a[] = {1, 2, 3, 4};
    S b[] = {0, 0, 0, 0};
    std::ranges::copy(std::make_reverse_iterator(wrap_n_times<InCount>(Iter(a + 4))),
                      std::make_reverse_iterator(wrap_n_times<InCount>(Iter(a))),
                      std::make_reverse_iterator(wrap_n_times<OutCount>(Iter(b + 4))));
    assert(std::equal(a, a + 4, b));
  }
  {
    S a[] = {1, 2, 3, 4};
    S b[] = {0, 0, 0, 0};
    auto range = std::ranges::subrange(wrap_n_times<InCount>(std::make_reverse_iterator(Iter(a + 4))),
                                       wrap_n_times<InCount>(std::make_reverse_iterator(Iter(a))));
    std::ranges::copy(range, std::make_reverse_iterator(wrap_n_times<OutCount>(Iter(b + 4))));
    assert(std::equal(a, a + 4, b));
  }
}

template <size_t InCount, size_t OutCount>
constexpr void test_normal_reverse() {
  test_normal<InCount, OutCount, S*>();
  test_normal<InCount, OutCount, NotIncrementableIt<S>>();
  test_reverse<InCount, OutCount, S*>();
  test_reverse<InCount, OutCount, NotIncrementableIt<S>>();
}

template <size_t InCount>
constexpr void test_out_count() {
  test_normal_reverse<InCount, 0>();
  test_normal_reverse<InCount, 2>();
  test_normal_reverse<InCount, 4>();
  test_normal_reverse<InCount, 6>();
  test_normal_reverse<InCount, 8>();
}

constexpr bool test() {
  test_out_count<0>();
  test_out_count<2>();
  test_out_count<4>();
  test_out_count<6>();
  test_out_count<8>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
