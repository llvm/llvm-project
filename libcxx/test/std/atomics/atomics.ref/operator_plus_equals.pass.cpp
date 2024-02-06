//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// integral-type operator+=(integral-type) const noexcept;
// floating-point-type operator+=(floating-point-type) const noexcept;
// T* operator+=(difference_type) const noexcept;

#include <atomic>
#include <concepts>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
concept has_operator_plus_equals = requires { std::declval<T const>() += std::declval<T>(); };

static_assert(!has_operator_plus_equals<std::atomic_ref<bool>>);
struct X {
  int i;
  X(int ii) noexcept : i(ii) {}
  bool operator==(X o) const { return i == o.i; }
};
static_assert(!has_operator_plus_equals<std::atomic_ref<X>>);

template <typename T>
void test_arithmetic() {
  T x(T(1));
  std::atomic_ref<T> const a(x);

  std::same_as<T> auto y = (a += T(2));
  assert(y == T(3));
  assert(x == T(3));
  ASSERT_NOEXCEPT(a += T(0));
}

template <typename T>
void test_pointer() {
  using U = std::remove_pointer_t<T>;
  U t[9]  = {};
  T p{&t[1]};
  std::atomic_ref<T> const a(p);

  std::same_as<T> auto y = (a += 2);
  assert(y == &t[3]);
  assert(a == &t[3]);
  ASSERT_NOEXCEPT(a += 0);
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
