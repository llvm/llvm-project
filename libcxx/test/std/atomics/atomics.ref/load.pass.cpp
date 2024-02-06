//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// T load(memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <concepts>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
void test_load() {
  T x(T(1));
  std::atomic_ref<T> const a(x);

  {
    std::same_as<T> auto y = a.load();
    assert(y == T(1));
    ASSERT_NOEXCEPT(a.load());
  }

  {
    std::same_as<T> auto y = a.load(std::memory_order_seq_cst);
    assert(y == T(1));
    ASSERT_NOEXCEPT(a.load(std::memory_order_seq_cst));
  }
}

void test() {
  test_load<int>();

  test_load<float>();

  test_load<int*>();

  struct X {
    int i;
    X(int ii) noexcept : i(ii) {}
    bool operator==(X o) const { return i == o.i; }
  };
  test_load<X>();
}

int main(int, char**) {
  test();
  return 0;
}
