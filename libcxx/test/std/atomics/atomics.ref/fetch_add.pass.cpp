//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// integral-type fetch_add(integral-type, memory_order = memory_order::seq_cst) const noexcept;
// floating-point-type fetch_add(floating-point-type, memory_order = memory_order::seq_cst) const noexcept;
// T* fetch_add(difference_type, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
concept has_fetch_add = requires {
  std::declval<T const>().fetch_add(std::declval<T>());
  std::declval<T const>().fetch_add(std::declval<T>(), std::declval<std::memory_order>());
};

static_assert(!has_fetch_add<std::atomic_ref<bool>>);
struct X {
  int i;
  X(int ii) noexcept : i(ii) {}
  bool operator==(X o) const { return i == o.i; }
};
static_assert(!has_fetch_add<std::atomic_ref<X>>);

template <typename T>
void test_arithmetic() {
  T x(T(1));
  std::atomic_ref<T> a(x);

  assert(a.fetch_add(T(2)) == T(1));
  assert(x == T(3));
  ASSERT_NOEXCEPT(a.fetch_add(T(0)));

  assert(a.fetch_add(T(4), std::memory_order_relaxed) == T(3));
  assert(x == T(7));
  ASSERT_NOEXCEPT(a.fetch_add(T(0), std::memory_order_relaxed));
}

template <typename T>
void test_pointer() {
  using X = std::remove_pointer_t<T>;
  X t[9]  = {};
  T p{&t[1]};
  std::atomic_ref<T> a(p);

  assert(a.fetch_add(2) == &t[1]);
  assert(a == &t[3]);
  ASSERT_NOEXCEPT(a.fetch_add(0));

  assert(a.fetch_add(4, std::memory_order_relaxed) == &t[3]);
  assert(a == &t[7]);
  ASSERT_NOEXCEPT(a.fetch_add(0, std::memory_order_relaxed));
}

void test() {
  test_arithmetic<int>();
  test_arithmetic<float>();

  test_pointer<int*>();
  test_pointer<const int*>();
}

int main(int, char**) {
  test();
  return 0;
}
