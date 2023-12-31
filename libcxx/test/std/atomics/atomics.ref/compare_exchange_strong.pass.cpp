//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// bool compare_exchange_strong(T&, T, memory_order, memory_order) const noexcept;
// bool compare_exchange_strong(T&, T, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
void test_compare_exchange_strong() {
  {
    T x(T(1));
    std::atomic_ref<T> a(x);

    T t(T(1));
    assert(a.compare_exchange_strong(t, T(2)) == true);
    assert(a == T(2));
    assert(t == T(1));
    assert(a.compare_exchange_strong(t, T(3)) == false);
    assert(a == T(2));
    assert(t == T(2));

    ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2)));
  }
  {
    T x(T(1));
    std::atomic_ref<T> a(x);

    T t(T(1));
    assert(a.compare_exchange_strong(t, T(2), std::memory_order_seq_cst) == true);
    assert(a == T(2));
    assert(t == T(1));
    assert(a.compare_exchange_strong(t, T(3), std::memory_order_seq_cst) == false);
    assert(a == T(2));
    assert(t == T(2));

    ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2), std::memory_order_seq_cst));
  }
  {
    T x(T(1));
    std::atomic_ref<T> a(x);

    T t(T(1));
    assert(a.compare_exchange_strong(t, T(2), std::memory_order_release, std::memory_order_relaxed) == true);
    assert(a == T(2));
    assert(t == T(1));
    assert(a.compare_exchange_strong(t, T(3), std::memory_order_release, std::memory_order_relaxed) == false);
    assert(a == T(2));
    assert(t == T(2));

    ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2), std::memory_order_release, std::memory_order_relaxed));
  }
}

void test() {
  test_compare_exchange_strong<int>();

  test_compare_exchange_strong<float>();

  test_compare_exchange_strong<int*>();

  struct X {
    int i;
    X(int ii) noexcept : i(ii) {}
    bool operator==(X o) const { return i == o.i; }
  };
  test_compare_exchange_strong<X>();
}

int main(int, char**) {
  test();
  return 0;
}
