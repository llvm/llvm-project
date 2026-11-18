//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_SEQUENCES_INPLACE_VECTOR_COMMON_H
#define TEST_STD_CONTAINERS_SEQUENCES_INPLACE_VECTOR_COMMON_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <inplace_vector>
#include <initializer_list>
#include <new>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "test_macros.h"

// inplace_vector<T,N> usage for non-trivial T in constant expressions is gated by
// implementations of P3074R7 (trivial unions), inplace_vector<T, N>
#ifdef __cpp_trivial_union
#  define TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR 1
#else
#  define TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR 0
#endif

template <class T>
using InplaceVector = std::inplace_vector<T, 32>;

template <class T>
using SmallInplaceVector = std::inplace_vector<T, 8>;

template <class T, class Expected>
constexpr void assert_inplace_vector_equal(const T& c, const Expected& expected) {
  assert(c.size() == std::ranges::size(expected));
  assert(std::ranges::equal(c, expected));
}

template <class C, class T>
constexpr void assert_inplace_vector_equal(const C& c, std::initializer_list<T> expected) {
  assert(c.size() == expected.size());
  assert(std::ranges::equal(c, expected));
}

template <class T, std::size_t N>
constexpr std::inplace_vector<T, N> make_inplace_vector(std::initializer_list<T> il) {
  std::inplace_vector<T, N> result;
  result.insert(result.end(), il.begin(), il.end());
  return result;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct ThrowingValue {
  inline static int alive       = 0;
  inline static int throw_after = -1;

  int value = 0;

  static void reset() {
    alive       = 0;
    throw_after = -1;
  }

  static void maybe_throw() {
    if (throw_after == 0)
      throw 1;
    if (throw_after > 0)
      --throw_after;
  }

  ThrowingValue() {
    maybe_throw();
    ++alive;
  }

  explicit ThrowingValue(int v) : value(v) {
    maybe_throw();
    ++alive;
  }

  ThrowingValue(const ThrowingValue& other) : value(other.value) {
    maybe_throw();
    ++alive;
  }

  ThrowingValue(ThrowingValue&& other) : value(other.value) {
    maybe_throw();
    ++alive;
  }

  ThrowingValue& operator=(const ThrowingValue& other) {
    maybe_throw();
    value = other.value;
    return *this;
  }

  ThrowingValue& operator=(ThrowingValue&& other) {
    maybe_throw();
    value = other.value;
    return *this;
  }

  ~ThrowingValue() { --alive; }

  friend bool operator==(const ThrowingValue& lhs, const ThrowingValue& rhs) { return lhs.value == rhs.value; }
};

template <class T, class IterCat>
struct throwing_iterator {
  using iterator_category = IterCat;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using reference         = T&;
  using pointer           = T*;

  int i_;
  T* ptr_;

  throwing_iterator() : i_(0), ptr_(nullptr) {}

  throwing_iterator(T* ptr, int i = 0) : i_(i), ptr_(ptr) {}

  reference operator*() const {
    if (i_ == 1)
      throw 1;
    return *ptr_;
  }

  friend bool operator==(const throwing_iterator& lhs, const throwing_iterator& rhs) { return lhs.i_ == rhs.i_; }
  friend bool operator!=(const throwing_iterator& lhs, const throwing_iterator& rhs) { return !(lhs == rhs); }

  throwing_iterator& operator++() {
    ++i_;
    ++ptr_;
    return *this;
  }

  throwing_iterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }
};

template <class Func>
void assert_throws_bad_alloc(Func func) {
  try {
    func();
    assert(false);
  } catch (const std::bad_alloc&) {
  }
}
#endif

#endif // TEST_STD_CONTAINERS_SEQUENCES_INPLACE_VECTOR_COMMON_H
