//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// ADDITIONAL_COMPILE_FLAGS(has-latomic): -latomic

// floating-point-type fetch_add(floating-point-type,
//                               memory_order = memory_order::seq_cst) volatile noexcept;
// floating-point-type fetch_add(floating-point-type,
//                               memory_order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <thread>
#include <type_traits>
#include <vector>

#include "test_helper.h"
#include "test_macros.h"

template <class T>
concept HasVolatileFetchAdd = requires(volatile std::atomic<T>& a, T t) { a.fetch_add(t); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  static_assert(HasVolatileFetchAdd<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().fetch_add(T(0))));

  // fetch_add concurrent
  {
    constexpr auto number_of_threads = 4;
    constexpr auto loop              = 1000;

    MaybeVolatile<std::atomic<T>> at;

    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);
    for (auto i = 0; i < number_of_threads; ++i) {
      threads.emplace_back([&at]() {
        for (auto j = 0; j < loop; ++j) {
          at.fetch_add(T(1.234), std::memory_order::relaxed);
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }

    const auto times = [](T t, int n) {
      T res(0);
      for (auto i = 0; i < n; ++i) {
        res += t;
      }
      return res;
    };

    assert(at.load() == times(1.234, number_of_threads * loop));
  }
}

template <class T>
void test() {
  test_impl<T>();
  if constexpr (std::atomic<T>::is_always_lock_free) {
    test_impl<T, std::add_volatile_t>();
  }
}

int main(int, char**) {
  test<double>();

  return 0;
}
