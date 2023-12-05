//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template <class It, class End>
// constexpr explicit(Extent != dynamic_extent) span(It first, End last);
// Requires: [first, last) shall be a valid range.
//   If Extent is not equal to dynamic_extent, then last - first shall be equal to Extent.
// Throws: When and what last - first throws.

#include <array>
#include <span>
#include <cassert>
#include <utility>

#include "assert_macros.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Sentinel>
constexpr bool test_ctor() {
  T val[2] = {};
  auto s1  = std::span<T>(std::begin(val), Sentinel(std::end(val)));
  auto s2  = std::span<T, 2>(std::begin(val), Sentinel(std::end(val)));
  assert(s1.data() == std::data(val) && s1.size() == std::size(val));
  assert(s2.data() == std::data(val) && s2.size() == std::size(val));
  return true;
}

template <std::size_t Extent>
constexpr void test_constructibility() {
  static_assert(std::is_constructible_v<std::span<int, Extent>, int*, int*>);
  static_assert(!std::is_constructible_v<std::span<int, Extent>, const int*, const int*>);
  static_assert(!std::is_constructible_v<std::span<int, Extent>, volatile int*, volatile int*>);
  static_assert(std::is_constructible_v<std::span<const int, Extent>, int*, int*>);
  static_assert(std::is_constructible_v<std::span<const int, Extent>, const int*, const int*>);
  static_assert(!std::is_constructible_v<std::span<const int, Extent>, volatile int*, volatile int*>);
  static_assert(std::is_constructible_v<std::span<volatile int, Extent>, int*, int*>);
  static_assert(!std::is_constructible_v<std::span<volatile int, Extent>, const int*, const int*>);
  static_assert(std::is_constructible_v<std::span<volatile int, Extent>, volatile int*, volatile int*>);
  static_assert(!std::is_constructible_v<std::span<int, Extent>, int*, float*>); // types wrong
}

constexpr bool test() {
  test_constructibility<std::dynamic_extent>();
  test_constructibility<3>();
  struct A {};
  assert((test_ctor<int, int*>()));
  assert((test_ctor<int, sized_sentinel<int*>>()));
  assert((test_ctor<A, A*>()));
  assert((test_ctor<A, sized_sentinel<A*>>()));
  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
// A stripped down contiguous iterator that throws when using operator-.
template <class It>
class throw_operator_minus {
  It it_;

public:
  typedef std::contiguous_iterator_tag iterator_category;
  typedef typename std::iterator_traits<It>::value_type value_type;
  typedef typename std::iterator_traits<It>::difference_type difference_type;
  typedef It pointer;
  typedef typename std::iterator_traits<It>::reference reference;
  typedef std::remove_reference_t<reference> element_type;

  throw_operator_minus() : it_() {}
  explicit throw_operator_minus(It it) : it_(it) {}

  reference operator*() const { return *it_; }
  pointer operator->() const { return it_; }
  reference operator[](difference_type n) const { return it_[n]; }

  throw_operator_minus& operator++() {
    ++it_;
    return *this;
  }
  throw_operator_minus& operator--() {
    --it_;
    return *this;
  }
  throw_operator_minus operator++(int) { return throw_operator_minus(it_++); }
  throw_operator_minus operator--(int) { return throw_operator_minus(it_--); }

  throw_operator_minus& operator+=(difference_type n) {
    it_ += n;
    return *this;
  }
  throw_operator_minus& operator-=(difference_type n) {
    it_ -= n;
    return *this;
  }
  friend throw_operator_minus operator+(throw_operator_minus x, difference_type n) {
    x += n;
    return x;
  }
  friend throw_operator_minus operator+(difference_type n, throw_operator_minus x) {
    x += n;
    return x;
  }
  friend throw_operator_minus operator-(throw_operator_minus x, difference_type n) {
    x -= n;
    return x;
  }
  friend difference_type operator-(throw_operator_minus, throw_operator_minus) { throw 42; };

  friend bool operator==(const throw_operator_minus& x, const throw_operator_minus& y) { return x.it_ == y.it_; }
  friend auto operator<=>(const throw_operator_minus& x, const throw_operator_minus& y) { return x.it_ <=> y.it_; }
};

template <class It>
throw_operator_minus(It) -> throw_operator_minus<It>;

void test_exceptions() {
  std::array a{42};
  TEST_VALIDATE_EXCEPTION(
      int,
      [](int i) { assert(i == 42); },
      (std::span<int>{throw_operator_minus{a.begin()}, throw_operator_minus{a.end()}}));
  TEST_VALIDATE_EXCEPTION(
      int,
      [](int i) { assert(i == 42); },
      (std::span<int, 1>{throw_operator_minus{a.begin()}, throw_operator_minus{a.end()}}));
}
#endif // TEST_HAS_NO_EXCEPTIONS

int main(int, char**) {
  test();
#ifndef TEST_HAS_NO_EXCEPTIONS
  test_exceptions();
#endif
  static_assert(test());

  return 0;
}
