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

// bool compare_exchange_strong(T& expected, T desired, memory_order success, memory_order failure) const noexcept;
//
// Preconditions: failure is memory_order::relaxed, memory_order::consume, memory_order::acquire, or memory_order::seq_cst.

#include <atomic>

#include "atomic_helpers.h"
#include "check_assertion.h"

template <typename T>
struct TestCompareExchangeStrongInvalidMemoryOrder {
  void operator()() const {
    { // no assertion should trigger here
      T x(T(1));
      std::atomic_ref<T> const a(x);
      T t(T(2));
      a.compare_exchange_strong(t, T(3), std::memory_order_relaxed, std::memory_order_relaxed);
    }

    TEST_LIBCPP_ASSERT_FAILURE(
        ([] {
          T x(T(1));
          std::atomic_ref<T> const a(x);
          T t(T(2));
          a.compare_exchange_strong(t, T(3), std::memory_order_relaxed, std::memory_order_release);
        }()),
        "atomic_ref: failure memory order argument to strong atomic compare-and-exchange operation is invalid");

    TEST_LIBCPP_ASSERT_FAILURE(
        ([] {
          T x(T(1));
          std::atomic_ref<T> const a(x);
          T t(T(2));
          a.compare_exchange_strong(t, T(3), std::memory_order_relaxed, std::memory_order_acq_rel);
        }()),
        "atomic_ref: failure memory order argument to strong atomic compare-and-exchange operation is invalid");
  }
};

int main(int, char**) {
  TestEachAtomicType<TestCompareExchangeStrongInvalidMemoryOrder>()();
  return 0;
}
