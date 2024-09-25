//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-threads
// XFAIL: availability-synchronization_library-missing
// XFAIL: !has-64-bit-atomics
// XFAIL: !has-1024-bit-atomics

// MSVC warning C4310: cast truncates constant value
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4310

// void wait(T, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "atomic_helpers.h"
#include "make_test_thread.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
struct TestWait {
  void operator()() const {
    {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      assert(a.load() == T(1));
      a.wait(T(0));
      std::thread t1 = support::make_test_thread([&]() {
        a.store(T(3));
        a.notify_one();
      });
      a.wait(T(1));
      assert(a.load() == T(3));
      t1.join();
      ASSERT_NOEXCEPT(a.wait(T(0)));

      assert(a.load() == T(3));
      a.wait(T(0), std::memory_order_seq_cst);
      std::thread t2 = support::make_test_thread([&]() {
        a.store(T(5));
        a.notify_one();
      });
      a.wait(T(3), std::memory_order_seq_cst);
      assert(a.load() == T(5));
      t2.join();
      ASSERT_NOEXCEPT(a.wait(T(0), std::memory_order_seq_cst));
    }

    // memory_order::acquire
    {
      auto store = [](std::atomic_ref<T> const& x, T, T new_val) { x.store(new_val, std::memory_order::release); };
      auto load  = [](std::atomic_ref<T> const& x) {
        auto result = x.load(std::memory_order::relaxed);
        x.wait(T(255), std::memory_order::acquire);
        return result;
      };
      test_acquire_release<T>(store, load);
    }

    // memory_order::seq_cst
    {
      auto store       = [](std::atomic_ref<T> const& x, T, T new_val) { x.store(new_val); };
      auto load_no_arg = [](std::atomic_ref<T> const& x) {
        auto result = x.load(std::memory_order::relaxed);
        x.wait(T(255));
        return result;
      };
      auto load_with_order = [](std::atomic_ref<T> const& x) {
        auto result = x.load(std::memory_order::relaxed);
        x.wait(T(255), std::memory_order::seq_cst);
        return result;
      };
      test_seq_cst<T>(store, load_no_arg);
      test_seq_cst<T>(store, load_with_order);
    }
  }
};

int main(int, char**) {
  TestEachAtomicType<TestWait>()();
  return 0;
}
