//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// integral-type operator&=(integral-type) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
concept has_bitwise_and_assign = requires { std::declval<T const>() &= std::declval<T>(); };

static_assert(!has_bitwise_and_assign<std::atomic_ref<float>>);
static_assert(!has_bitwise_and_assign<std::atomic_ref<int*>>);
static_assert(!has_bitwise_and_assign<std::atomic_ref<const int*>>);
static_assert(!has_bitwise_and_assign<std::atomic_ref<bool>>);
struct X {
  int i;
  X(int ii) noexcept : i(ii) {}
  bool operator==(X o) const { return i == o.i; }
};
static_assert(!has_bitwise_and_assign<std::atomic_ref<X>>);

template <typename T>
void test_integral() {
  T x(T(1));
  std::atomic_ref<T> const a(x);

  assert((a &= T(2)) == T(0));
  assert(x == T(0));
  ASSERT_NOEXCEPT(a &= T(0));
}

void test() { test_integral<int>(); }

int main(int, char**) {
  test();
  return 0;
}
