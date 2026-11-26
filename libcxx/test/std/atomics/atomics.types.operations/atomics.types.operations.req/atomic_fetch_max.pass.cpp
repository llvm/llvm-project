//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// XFAIL: !has-64-bit-atomics

// <atomic>

// template<class T>
//   T atomic_fetch_max(volatile atomic<T>*, typename atomic<T>::value_type) noexcept;
// template<class T>
//   constexpr T atomic_fetch_max(atomic<T>*, typename atomic<T>::value_type) noexcept;

#include <algorithm>
#include <atomic>
#include <cassert>
#include <type_traits>
#include <vector>

#include "atomic_fetch_max_helper.h"
#include "atomic_helpers.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_THREADS
#  include <thread>

#  include "make_test_thread.h"
#endif

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  {
    MaybeVolatile<std::atomic<T>> a;
    auto load  = [&]() { return a.load(); };
    auto store = [&](T val) { a.store(val); };
    auto max   = [&](T val, auto) { return std::atomic_fetch_max(&a, val); };
    ASSERT_NOEXCEPT(std::atomic_fetch_max(&a, T(0)));
    test_fetch_max_integral<T>(load, store, max);
  }

#ifndef TEST_HAS_NO_THREADS
  // Concurrent stress test: many threads calling atomic_fetch_max should leave
  // the atomic at the global maximum value seen.
  {
    constexpr auto number_of_threads = 4;
    constexpr auto loop              = 30;

    MaybeVolatile<std::atomic<T>> at(0);

    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);
    for (auto i = 0; i < number_of_threads; ++i) {
      threads.push_back(support::make_test_thread([&at, i]() {
        for (auto j = 0; j < loop; ++j) {
          std::atomic_fetch_max(&at, T(i * loop + j));
        }
      }));
    }

    for (auto& thread : threads) {
      thread.join();
    }

    const auto times = [](int n) {
      T res(0);
      for (auto i = 0; i < n; ++i) {
        res = std::max<T>(res, i);
      }
      return res;
    };

    assert(at.load() == times(number_of_threads * loop));
  }
#endif
}

template <class T>
struct TestFn {
  void operator()() const {
    test_impl<T>();
    if constexpr (std::atomic<T>::is_always_lock_free) {
      test_impl<T, std::add_volatile_t>();
    }
  }
};

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_pointer_impl() {
  T arr[5];
  T* p0 = &arr[0];
  T* p2 = &arr[2];
  T* p4 = &arr[4];

  MaybeVolatile<std::atomic<T*>> a;
  auto load  = [&]() { return a.load(); };
  auto store = [&](T* val) { a.store(val); };
  auto max   = [&](T* val, auto) { return std::atomic_fetch_max(&a, val); };
  ASSERT_NOEXCEPT(std::atomic_fetch_max(&a, p0));
  test_fetch_max_pointer<T>(p0, p2, p4, load, store, max);
}

template <class T>
void test_pointer() {
  test_pointer_impl<T>();
  if constexpr (std::atomic<T*>::is_always_lock_free) {
    test_pointer_impl<T, std::add_volatile_t>();
  }
}

int main(int, char**) {
  TestEachIntegralType<TestFn>()();
  test_pointer<int>();
  test_pointer<char>();

  return 0;
}
