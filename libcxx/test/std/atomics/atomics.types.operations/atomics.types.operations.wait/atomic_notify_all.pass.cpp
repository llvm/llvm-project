//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03
// XFAIL: !has-1024-bit-atomics

// Until we drop support for the synchronization library in C++11/14/17
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// XFAIL: availability-synchronization_library-missing

// <atomic>

// template<class T>
//     void
//     atomic_notify_all(volatile atomic<T>*) noexcept;
//
// template<class T>
//     void
//     atomic_notify_all(atomic<T>*) noexcept;

#include <atomic>
#include <type_traits>
#include <cassert>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"
#include "atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    typedef std::atomic<T> A;

    {
      A a(T(1));
      static_assert(noexcept(std::atomic_notify_all(&a)), "");

      std::atomic<bool> is_ready[2];
      is_ready[0] = false;
      is_ready[1] = false;
      auto f      = [&](int index) {
        assert(std::atomic_load(&a) == T(1));
        is_ready[index].store(true);

        std::atomic_wait(&a, T(1));
        assert(std::atomic_load(&a) == T(3));
      };
      std::thread t1 = support::make_test_thread(f, /*index=*/0);
      std::thread t2 = support::make_test_thread(f, /*index=*/1);

      while (!is_ready[0] || !is_ready[1]) {
        // Spin
      }
      std::atomic_store(&a, T(3));
      std::atomic_notify_all(&a);
      t1.join();
      t2.join();
    }
    {
      volatile A a(T(2));
      static_assert(noexcept(std::atomic_notify_all(&a)), "");

      std::atomic<bool> is_ready[2];
      is_ready[0] = false;
      is_ready[1] = false;
      auto f      = [&](int index) {
        assert(std::atomic_load(&a) == T(2));
        is_ready[index].store(true);

        std::atomic_wait(&a, T(2));
        assert(std::atomic_load(&a) == T(4));
      };
      std::thread t1 = support::make_test_thread(f, /*index=*/0);
      std::thread t2 = support::make_test_thread(f, /*index=*/1);

      while (!is_ready[0] || !is_ready[1]) {
        // Spin
      }
      std::atomic_store(&a, T(4));
      std::atomic_notify_all(&a);
      t1.join();
      t2.join();
    }
  }
};

int main(int, char**) {
  TestEachAtomicType<TestFn>()();

  return 0;
}
