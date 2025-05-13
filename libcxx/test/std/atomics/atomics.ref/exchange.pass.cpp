//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// XFAIL: !has-1024-bit-atomics

// T exchange(T, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
struct TestExchange {
  void operator()() const {
    {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      {
        std::same_as<T> decltype(auto) y = a.exchange(T(2));
        assert(y == T(1));
        ASSERT_NOEXCEPT(a.exchange(T(2)));
      }

      {
        std::same_as<T> decltype(auto) y = a.exchange(T(3), std::memory_order_seq_cst);
        assert(y == T(2));
        ASSERT_NOEXCEPT(a.exchange(T(3), std::memory_order_seq_cst));
      }
    }

    // memory_order::release
    {
      auto exchange = [](std::atomic_ref<T> const& x, T, T new_val) {
        x.exchange(new_val, std::memory_order::release);
      };
      auto load = [](std::atomic_ref<T> const& x) { return x.load(std::memory_order::acquire); };
      test_acquire_release<T>(exchange, load);
    }

    // memory_order::seq_cst
    {
      auto exchange_no_arg     = [](std::atomic_ref<T> const& x, T, T new_val) { x.exchange(new_val); };
      auto exchange_with_order = [](std::atomic_ref<T> const& x, T, T new_val) {
        x.exchange(new_val, std::memory_order::seq_cst);
      };
      auto load = [](std::atomic_ref<T> const& x) { return x.load(); };
      test_seq_cst<T>(exchange_no_arg, load);
      test_seq_cst<T>(exchange_with_order, load);
    }
  }
};

int main(int, char**) {
  TestEachAtomicType<TestExchange>()();
  return 0;
}
