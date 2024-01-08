//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -Wno-user-defined-warnings

// <atomic>

// bool compare_exchange_weak(T& expected, T desired, memory_order success, memory_order failure) const noexcept;
//
// Preconditions: failure is memory_order::relaxed, memory_order::consume, memory_order::acquire, or memory_order::seq_cst.

#include <atomic>

#include "check_assertion.h"

template <typename T>
void test_compare_exchange_weak_invalid_memory_order() {
  {
    T x(T(1));
    std::atomic_ref<T> a(x);
    T t(T(2));
    a.compare_exchange_weak(t, T(3), std::memory_order_relaxed, std::memory_order_relaxed);
  }

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        T x(T(1));
        std::atomic_ref<T> a(x);
        T t(T(2));
        a.compare_exchange_weak(t, T(3), std::memory_order_relaxed, std::memory_order_release);
      }()),
      "memory order argument to weak atomic compare-and-exchange operation is invalid");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        T x(T(1));
        std::atomic_ref<T> a(x);
        T t(T(2));
        a.compare_exchange_weak(t, T(3), std::memory_order_relaxed, std::memory_order_acq_rel);
      }()),
      "memory order argument to weak atomic compare-and-exchange operation is invalid");
}

int main(int, char**) {
  test_compare_exchange_weak_invalid_memory_order<int>();
  test_compare_exchange_weak_invalid_memory_order<float>();
  test_compare_exchange_weak_invalid_memory_order<int*>();
  struct X {
    int i;
    X(int ii) noexcept : i(ii) {}
    bool operator==(X o) const { return i == o.i; }
  };
  test_compare_exchange_weak_invalid_memory_order<X>();

  return 0;
}
