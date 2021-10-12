//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// XFAIL: c++03
// XFAIL: !non-lockfree-atomics

// This test requires the dylib support introduced in D68480, which shipped in
// macOS 11.0.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}

// <atomic>

// template<class T>
//     void
//     atomic_notify_all(volatile atomic<T>*);
//
// template<class T>
//     void
//     atomic_notify_all(atomic<T>*);

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
      auto f = [&]() {
        assert(std::atomic_load(&a) == T(1));
        std::atomic_wait(&a, T(1));
        assert(std::atomic_load(&a) == T(3));
      };
      std::thread t1 = support::make_test_thread(f);
      std::thread t2 = support::make_test_thread(f);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      std::atomic_store(&a, T(3));
      std::atomic_notify_all(&a);
      t1.join();
      t2.join();
    }
    {
      volatile A a(T(2));
      auto f = [&]() {
        assert(std::atomic_load(&a) == T(2));
        std::atomic_wait(&a, T(2));
        assert(std::atomic_load(&a) == T(4));
      };
      std::thread t1 = support::make_test_thread(f);
      std::thread t2 = support::make_test_thread(f);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

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
