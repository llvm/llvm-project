//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

//  floating-point-type load(memory_order = memory_order::seq_cst) volatile noexcept;
//  floating-point-type load(memory_order = memory_order::seq_cst) noexcept;

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
concept HasVolatileLoad = requires(volatile std::atomic<T>& a, T t) { a.load(); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  // Uncomment the test after P1831R1 is implemented
  // static_assert(HasVolatileLoad<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().load()));

  // load
  {
    MaybeVolatile<std::atomic<T>> a(T(3.1));
    a.store(T(1.2));
    std::same_as<T> decltype(auto) r = a.load(std::memory_order::relaxed);
    assert(r == T(1.2));
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
          at.store(T(i));
        }
      }));
    }

    while (at.load(std::memory_order::relaxed) == T(-1.0)) {
      std::this_thread::yield();
    }

    for (auto i = 0; i < loop; ++i) {
      auto r = at.load(std::memory_order_relaxed);
      assert(std::ranges::any_of(std::views::iota(0, number_of_threads), [r](auto j) { return r == T(j); }));
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  // memory_order::consume
  {
    std::unique_ptr<T> p = std::make_unique<T>(T(0.0));
    MaybeVolatile<std::atomic<T>> at(T(0.0));

    constexpr auto number_of_threads = 8;
    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);

    for (auto i = 0; i < number_of_threads; ++i) {
      threads.push_back(support::make_test_thread([&at, &p] {
        while (at.load(std::memory_order::consume) == T(0.0)) {
          std::this_thread::yield();
        }
        assert(*p == T(1.0)); // the write from other thread should be visible
      }));
    }

    *p = T(1.0);
    at.store(*p, std::memory_order_release);

    for (auto& thread : threads) {
      thread.join();
    }
  }
#endif

  // memory_order::acquire
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val, std::memory_order::release); };
    auto load  = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(std::memory_order::acquire); };
    test_acquire_release<T, MaybeVolatile>(store, load);
  }

  // memory_order::seq_cst
  {
    auto store           = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val); };
    auto load_no_arg     = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(); };
    auto load_with_order = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(std::memory_order::seq_cst); };
    test_seq_cst<T, MaybeVolatile>(store, load_no_arg);
    test_seq_cst<T, MaybeVolatile>(store, load_with_order);
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
