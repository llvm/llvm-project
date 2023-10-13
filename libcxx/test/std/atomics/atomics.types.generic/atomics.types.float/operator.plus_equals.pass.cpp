//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// ADDITIONAL_COMPILE_FLAGS(has-latomic): -latomic

// floating-point-type operator+=(floating-point-type) volatile noexcept;
// floating-point-type operator+=(floating-point-type) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <thread>
#include <type_traits>
#include <vector>

#include "test_helper.h"
#include "test_macros.h"

template <class T>
concept HasVolatilePlusEquals = requires(volatile std::atomic<T> a, T t) { a += t; };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void testImpl() {
  static_assert(HasVolatilePlusEquals<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>() += T(0)));

  // +=
  {
    MaybeVolatile<std::atomic<T>> a(3.1);
    std::same_as<T> decltype(auto) r = a += T(1.2);
    assert(r == T(3.1) + T(1.2));
    assert(a.load() == T(3.1) + T(1.2));
  }

  // += concurrent
  {
    constexpr auto number_of_threads = 4;
    constexpr auto loop              = 1000;

    MaybeVolatile<std::atomic<T>> at;

    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);
    for (auto i = 0; i < number_of_threads; ++i) {
      threads.emplace_back([&at]() {
        for (auto j = 0; j < loop; ++j) {
          at += T(1.234);
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

  // memory_order::seq_cst
  {
    auto plus_equals = [](MaybeVolatile<std::atomic<T>>& x, T old_value, T new_val) { x += (new_val - old_value); };
    auto load        = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(); };
    test_seq_cst<T, MaybeVolatile>(plus_equals, load);
  }
}

template <class T>
void test() {
  testImpl<T>();
  if constexpr (std::atomic<T>::is_always_lock_free) {
    testImpl<T, std::add_volatile_t>();
  }
}

int main(int, char**) {
  test<float>();
  test<double>();
  test<long double>();

  return 0;
}
