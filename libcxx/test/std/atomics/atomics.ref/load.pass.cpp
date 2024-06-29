//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// XFAIL: !has-1024-bit-atomics

// T load(memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <concepts>
#include <cassert>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
struct TestLoad {
  void operator()() const {
    T x(T(1));
    std::atomic_ref<T> const a(x);

    {
      std::same_as<T> decltype(auto) y = a.load();
      assert(y == T(1));
      ASSERT_NOEXCEPT(a.load());
    }

    {
      std::same_as<T> decltype(auto) y = a.load(std::memory_order_seq_cst);
      assert(y == T(1));
      ASSERT_NOEXCEPT(a.load(std::memory_order_seq_cst));
    }

    // memory_order::seq_cst
    {
      auto store           = [](std::atomic_ref<T> const& y, T, T new_val) { y.store(new_val); };
      auto load_no_arg     = [](std::atomic_ref<T> const& y) { return y.load(); };
      auto load_with_order = [](std::atomic_ref<T> const& y) { return y.load(std::memory_order::seq_cst); };
      test_seq_cst<T>(store, load_no_arg);
      test_seq_cst<T>(store, load_with_order);
    }

    // memory_order::release
    {
      auto store = [](std::atomic_ref<T> const& y, T, T new_val) { y.store(new_val, std::memory_order::release); };
      auto load  = [](std::atomic_ref<T> const& y) { return y.load(std::memory_order::acquire); };
      test_acquire_release<T>(store, load);
    }
  }
};

int main(int, char**) {
  TestEachAtomicType<TestLoad>()();
  return 0;
}
