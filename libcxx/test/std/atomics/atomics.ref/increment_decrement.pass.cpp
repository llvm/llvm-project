//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// integral-type operator++(int) const noexcept;
// integral-type operator--(int) const noexcept;
// integral-type operator++() const noexcept;
// integral-type operator--() const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
concept has_pre_increment_operator = requires { ++std::declval<T const>(); };

template <typename T>
concept has_post_increment_operator = requires { std::declval<T const>()++; };

template <typename T>
concept has_pre_decrement_operator = requires { --std::declval<T const>(); };

template <typename T>
concept has_post_decrement_operator = requires { std::declval<T const>()--; };

template <typename T>
constexpr bool does_not_have_increment_nor_decrement_operators() {
  return !has_pre_increment_operator<T> && !has_pre_decrement_operator<T> && !has_post_increment_operator<T> &&
         !has_post_decrement_operator<T>;
}

static_assert(does_not_have_increment_nor_decrement_operators<float>());
static_assert(does_not_have_increment_nor_decrement_operators<int*>());
static_assert(does_not_have_increment_nor_decrement_operators<const int*>());
static_assert(does_not_have_increment_nor_decrement_operators<bool>());
struct X {
  int i;
  X(int ii) noexcept : i(ii) {}
  bool operator==(X o) const { return i == o.i; }
};
static_assert(does_not_have_increment_nor_decrement_operators<X>());

template <typename T>
void test_integral() {
  T x(T(1));
  std::atomic_ref<T> const a(x);

  assert(++a == T(2));
  assert(x == T(2));
  ASSERT_NOEXCEPT(++a);

  assert(--a == T(1));
  assert(x == T(1));
  ASSERT_NOEXCEPT(--a);

  assert(a++ == T(1));
  assert(x == T(2));
  ASSERT_NOEXCEPT(++a);

  assert(a-- == T(2));
  assert(x == T(1));
  ASSERT_NOEXCEPT(--a);
}

void test() { test_integral<int>(); }

int main(int, char**) {
  test();
  return 0;
}
