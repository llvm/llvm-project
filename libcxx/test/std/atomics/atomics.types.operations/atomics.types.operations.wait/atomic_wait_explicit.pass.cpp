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
//     atomic_wait_explicit(const volatile atomic<T>*, atomic<T>::value_type,
//                          memory_order) noexcept;
//
// template<class T>
//     void
//     atomic_wait_explicit(const volatile atomic<T>*, atomic<T>::value_type,
//                          memory_order) noexcept;

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
      A t(T(1));
      static_assert(noexcept(std::atomic_wait_explicit(&t, T(0), std::memory_order_seq_cst)), "");
      assert(std::atomic_load(&t) == T(1));
      std::atomic_wait_explicit(&t, T(0), std::memory_order_seq_cst);
      std::thread t1 = support::make_test_thread([&]() {
        std::atomic_store(&t, T(3));
        std::atomic_notify_one(&t);
      });
      std::atomic_wait_explicit(&t, T(1), std::memory_order_seq_cst);
      assert(std::atomic_load(&t) == T(3));
      t1.join();
    }
    {
      volatile A vt(T(2));
      static_assert(noexcept(std::atomic_wait_explicit(&vt, T(0), std::memory_order_seq_cst)), "");
      assert(std::atomic_load(&vt) == T(2));
      std::atomic_wait_explicit(&vt, T(1), std::memory_order_seq_cst);
      std::thread t2 = support::make_test_thread([&]() {
        std::atomic_store(&vt, T(4));
        std::atomic_notify_one(&vt);
      });
      std::atomic_wait_explicit(&vt, T(2), std::memory_order_seq_cst);
      assert(std::atomic_load(&vt) == T(4));
      t2.join();
    }
  }
};

int main(int, char**) {
  TestEachAtomicType<TestFn>()();

  return 0;
}
