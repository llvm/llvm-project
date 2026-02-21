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
//   T atomic_fetch_min_explicit(volatile atomic<T>*, typename atomic<T>::value_type,
//                               memory_order) noexcept;
// template<class T>
//   constexpr T atomic_fetch_min_explicit(atomic<T>*, typename atomic<T>::value_type,
//                                         memory_order) noexcept;

#include <algorithm>
#include <atomic>
#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomic_helpers.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_THREADS
#  include <thread>

#  include "make_test_thread.h"
#endif

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  static_assert(noexcept(
      std::atomic_fetch_min_explicit(std::declval<MaybeVolatile<std::atomic<T>>*>(), T(0), std::memory_order_seq_cst)));

  // atomic_fetch_min_explicit
  {
    MaybeVolatile<std::atomic<T>> t(T(1));
    assert(std::atomic_fetch_min_explicit(&t, T(2), std::memory_order_seq_cst) == T(1));
    assert(t == T(1));
  }
  {
    MaybeVolatile<std::atomic<T>> t(T(3));
    std::same_as<T> decltype(auto) r = t.fetch_min(T(2), std::memory_order_seq_cst);
    assert(r == T(3));
    assert(t == T(2));
  }

#ifndef TEST_HAS_NO_THREADS
  // atomic_fetch_min_explicit concurrent
  {
    constexpr auto number_of_threads = 4;
    constexpr auto loop              = 30;

    MaybeVolatile<std::atomic<T>> at(number_of_threads * loop);

    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);
    for (auto i = 0; i < number_of_threads; ++i) {
      threads.push_back(support::make_test_thread([&at, i]() {
        for (auto j = loop - 1; j >= 0; --j) {
          std::atomic_fetch_min_explicit(&at, T(i * loop + j), std::memory_order_relaxed);
        }
      }));
    }

    for (auto& thread : threads) {
      thread.join();
    }

    const auto times = [](int n) {
      T res(n);
      for (auto i = n - 1; i >= 0; --i) {
        res = std::min<T>(res, i);
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

int main(int, char**) {
  TestEachIntegralType<TestFn>()();

  return 0;
}
