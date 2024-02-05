//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// operator T() const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
void test_convert() {
  T x(T(1));
  std::atomic_ref<T> const a(x);

  T converted = a;
  assert(converted == T(1));

  ASSERT_NOEXCEPT(T(a));
  static_assert(std::is_nothrow_convertible_v<std::atomic_ref<T>, T>);
}

void test() {
  test_convert<int>();

  test_convert<float>();

  test_convert<int*>();

  struct X {
    int i;
    X(int ii) noexcept : i(ii) {}
    bool operator==(X o) const { return i == o.i; }
  };
  test_convert<X>();
}

int main(int, char**) {
  test();
  return 0;
}
