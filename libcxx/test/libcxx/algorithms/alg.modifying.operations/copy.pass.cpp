//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: debug_level=1

// <algorithm>

// This test checks that std::copy forwards to memmove when appropriate.

#include <algorithm>
#include <cassert>
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

  constexpr NotIncrementableIt(T* i_) : i(i_) {}

  friend constexpr bool operator==(const NotIncrementableIt& lhs, const NotIncrementableIt& rhs) {
    return lhs.i == rhs.i;
  }

  constexpr T& operator*() { return *i; }
  constexpr T* operator->() { return i; }
  constexpr T* operator->() const { return i; }

  constexpr NotIncrementableIt& operator++() {
    assert(false);
    return *this;
  }

  constexpr NotIncrementableIt& operator--() {
    assert(false);
    return *this;
  }

  friend constexpr NotIncrementableIt operator+(const NotIncrementableIt& it, ptrdiff_t size) { return it.i + size; }
};

static_assert(std::__is_cpp17_contiguous_iterator<NotIncrementableIt<S>>::value);

template <class Iter>
constexpr void test_normal() {
  S a[] = {1, 2, 3, 4};
  S b[] = {0, 0, 0, 0};
  std::copy(Iter(a), Iter(a + 4), Iter(b));
  assert(std::equal(a, a + 4, b));
}

template <class Iter>
constexpr void test_reverse() {
  S a[] = {1, 2, 3, 4};
  S b[] = {0, 0, 0, 0};
  std::copy(std::make_reverse_iterator(Iter(a + 4)),
            std::make_reverse_iterator(Iter(a)),
            std::make_reverse_iterator(Iter(b + 4)));
}

template <class Iter>
constexpr void test_reverse_reverse() {
  S a[] = {1, 2, 3, 4};
  S b[] = {0, 0, 0, 0};
  std::copy(std::make_reverse_iterator(std::make_reverse_iterator(Iter(a))),
            std::make_reverse_iterator(std::make_reverse_iterator(Iter(a + 4))),
            std::make_reverse_iterator(std::make_reverse_iterator(Iter(b))));
}

constexpr bool test() {
  test_normal<S*>();
  test_normal<NotIncrementableIt<S>>();
  test_reverse<S*>();
  test_reverse<NotIncrementableIt<S>>();
  test_reverse_reverse<S*>();
  test_reverse_reverse<NotIncrementableIt<S>>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
}
