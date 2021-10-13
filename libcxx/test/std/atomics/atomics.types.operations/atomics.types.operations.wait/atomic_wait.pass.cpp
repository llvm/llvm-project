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
//     atomic_wait(const volatile atomic<T>*, atomic<T>::value_type);
//
// template<class T>
//     void
//     atomic_wait(const atomic<T>*, atomic<T>::value_type);

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
      assert(std::atomic_load(&t) == T(1));
      std::atomic_wait(&t, T(0));
      std::thread t1 = support::make_test_thread([&]() {
        std::atomic_store(&t, T(3));
        std::atomic_notify_one(&t);
      });
      std::atomic_wait(&t, T(1));
      assert(std::atomic_load(&t) == T(3));
      t1.join();
    }
    {
      volatile A vt(T(2));
      assert(std::atomic_load(&vt) == T(2));
      std::atomic_wait(&vt, T(1));
      std::thread t2 = support::make_test_thread([&]() {
        std::atomic_store(&vt, T(4));
        std::atomic_notify_one(&vt);
      });
      std::atomic_wait(&vt, T(2));
      assert(std::atomic_load(&vt) == T(4));
      t2.join();
    }
  }
};

int main(int, char**) {
  TestEachAtomicType<TestFn>()();

  return 0;
}
