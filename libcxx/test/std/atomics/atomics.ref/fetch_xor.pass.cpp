//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// integral-type fetch_xor(integral-type, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
concept has_fetch_xor = requires {
  std::declval<T const>().fetch_xor(std::declval<T>());
  std::declval<T const>().fetch_xor(std::declval<T>(), std::declval<std::memory_order>());
};

static_assert(!has_fetch_xor<std::atomic_ref<float>>);
static_assert(!has_fetch_xor<std::atomic_ref<int*>>);
static_assert(!has_fetch_xor<std::atomic_ref<const int*>>);
static_assert(!has_fetch_xor<std::atomic_ref<bool>>);
struct X {
  int i;
  X(int ii) noexcept : i(ii) {}
  bool operator==(X o) const { return i == o.i; }
};
static_assert(!has_fetch_xor<std::atomic_ref<X>>);

template <typename T>
void test_integral() {
  T x(T(1));
  std::atomic_ref<T> a(x);

  assert(a.fetch_xor(T(2)) == T(1));
  assert(x == T(3));
  ASSERT_NOEXCEPT(a.fetch_xor(T(0)));

  assert(a.fetch_xor(T(2), std::memory_order_relaxed) == T(3));
  assert(x == T(1));
  ASSERT_NOEXCEPT(a.fetch_xor(T(0), std::memory_order_relaxed));
}

void test() { test_integral<int>(); }

int main(int, char**) {
  test();
  return 0;
}
