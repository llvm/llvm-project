//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// XFAIL: !has-1024-bit-atomics

// void store(T, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
struct TestStore {
  void operator()() const {
    T x(T(1));
    std::atomic_ref<T> const a(x);

    a.store(T(2));
    assert(x == T(2));
    ASSERT_NOEXCEPT(a.store(T(1)));

    a.store(T(3), std::memory_order_seq_cst);
    assert(x == T(3));
    ASSERT_NOEXCEPT(a.store(T(0), std::memory_order_seq_cst));

    // TODO memory_order::relaxed

    // memory_order::seq_cst
    {
      auto store_no_arg     = [](std::atomic_ref<T> const& y, T, T new_val) { y.store(new_val); };
      auto store_with_order = [](std::atomic_ref<T> const& y, T, T new_val) {
        y.store(new_val, std::memory_order::seq_cst);
      };
      auto load = [](std::atomic_ref<T> const& y) { return y.load(); };
      test_seq_cst<T>(store_no_arg, load);
      test_seq_cst<T>(store_with_order, load);
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
  TestEachAtomicType<TestStore>()();
  return 0;
}
