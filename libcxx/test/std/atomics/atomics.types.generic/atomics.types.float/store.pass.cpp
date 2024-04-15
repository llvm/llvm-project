//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// void store(floating-point-type, memory_order = memory_order::seq_cst) volatile noexcept;
// void store(floating-point-type, memory_order = memory_order::seq_cst) noexcept;

#include <algorithm>
#include <atomic>
#include <cassert>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <vector>

#include "test_helper.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_THREADS
#  include "make_test_thread.h"
#  include <thread>
#endif

template <class T>
concept HasVolatileStore = requires(volatile std::atomic<T>& a, T t) { a.store(t); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  // Uncomment the test after P1831R1 is implemented
  // static_assert(HasVolatileStore<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().store(T(0))));

  // store
  {
    MaybeVolatile<std::atomic<T>> a(T(3.1));
    a.store(T(1.2), std::memory_order::relaxed);
    assert(a.load() == T(1.2));
  }

#ifndef TEST_HAS_NO_THREADS
  // memory_order::relaxed
  {
    constexpr auto number_of_threads = 4;
    constexpr auto loop              = 1000;

    MaybeVolatile<std::atomic<T>> at(T(-1.0));

    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);
    for (auto i = 0; i < number_of_threads; ++i) {
      threads.push_back(support::make_test_thread([&at, i]() {
        for (auto j = 0; j < loop; ++j) {
          at.store(T(i), std::memory_order_relaxed);
        }
      }));
    }

    while (at.load() == T(-1.0)) {
      std::this_thread::yield();
    }

    for (auto i = 0; i < loop; ++i) {
      auto r = at.load();
      assert(std::ranges::any_of(std::views::iota(0, number_of_threads), [r](auto j) { return r == T(j); }));
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
#endif

  // memory_order::release
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val, std::memory_order::release); };
    auto load  = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(std::memory_order::acquire); };
    test_acquire_release<T, MaybeVolatile>(store, load);
  }

  // memory_order::seq_cst
  {
    auto store_no_arg     = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val); };
    auto store_with_order = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) {
      x.store(new_val, std::memory_order::seq_cst);
    };
    auto load = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(); };
    test_seq_cst<T, MaybeVolatile>(store_no_arg, load);
    test_seq_cst<T, MaybeVolatile>(store_with_order, load);
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
  test<float>();
  test<double>();
  // TODO https://github.com/llvm/llvm-project/issues/47978
  // test<long double>();

  return 0;
}
