//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-hardening-mode=none || libcpp-hardening-mode=fast
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -Wno-user-defined-warnings

// <atomic>

// void wait(T old, memory_order order = memory_order::seq_cst) const noexcept;
//
// Preconditions: order is memory_order::relaxed, memory_order::consume, memory_order::acquire, or memory_order::seq_cst.

#include <atomic>

#include "check_assertion.h"

template <typename T>
void test_wait_invalid_memory_order() {
  {
    T x(T(1));
    std::atomic_ref<T> const a(x);
    a.wait(T(2), std::memory_order_relaxed);
  }

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        T x(T(1));
        std::atomic_ref<T> const a(x);
        a.wait(T(2), std::memory_order_release);
      }()),
      "atomic_ref: memory order argument to atomic wait operation is invalid");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        T x(T(1));
        std::atomic_ref<T> const a(x);
        a.wait(T(2), std::memory_order_acq_rel);
      }()),
      "atomic_ref: memory order argument to atomic wait operation is invalid");
}

int main(int, char**) {
  test_wait_invalid_memory_order<int>();
  test_wait_invalid_memory_order<float>();
  test_wait_invalid_memory_order<int*>();
  struct X {
    int i;
    X(int ii) noexcept : i(ii) {}
    bool operator==(X o) const { return i == o.i; }
  };
  test_wait_invalid_memory_order<X>();

  return 0;
}
