//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// Older versions of clang have a bug with atomic builtins affecting double and long double.
// Fixed by 5fdd0948.
// XFAIL: target=powerpc-ibm-{{.*}} && (clang-17 || clang-18)

// https://github.com/llvm/llvm-project/issues/72893
// XFAIL: target={{x86_64-.*}} && tsan

// floating-point-type fetch_sub(floating-point-type,
//                               memory_order = memory_order::seq_cst) volatile noexcept;
// floating-point-type fetch_sub(floating-point-type,
//                               memory_order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_helper.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_THREADS
#  include "make_test_thread.h"
#  include <thread>
#endif

template <class T>
concept HasVolatileFetchSub = requires(volatile std::atomic<T>& a, T t) { a.fetch_sub(t); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  static_assert(HasVolatileFetchSub<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().fetch_sub(T(0))));

  // fetch_sub
  {
    MaybeVolatile<std::atomic<T>> a(T(3.1));
    std::same_as<T> decltype(auto) r = a.fetch_sub(T(1.2), std::memory_order::relaxed);
    assert(r == T(3.1));
    assert(a.load() == T(3.1) - T(1.2));
  }

#ifndef TEST_HAS_NO_THREADS
  // fetch_sub concurrent
  {
    constexpr auto number_of_threads = 4;
    constexpr auto loop              = 1000;

    MaybeVolatile<std::atomic<T>> at;

    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);
    for (auto i = 0; i < number_of_threads; ++i) {
      threads.push_back(support::make_test_thread([&at]() {
        for (auto j = 0; j < loop; ++j) {
          at.fetch_sub(T(1.234), std::memory_order::relaxed);
        }
      }));
    }

    for (auto& thread : threads) {
      thread.join();
    }

    const auto accu_neg = [](T t, int n) {
      T res(0);
      for (auto i = 0; i < n; ++i) {
        res -= t;
      }
      return res;
    };

    assert(at.load() == accu_neg(T(1.234), number_of_threads * loop));
  }
#endif

  // memory_order::release
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T old_val, T new_val) {
      x.fetch_sub(old_val - new_val, std::memory_order::release);
    };
    auto load = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(std::memory_order::acquire); };
    test_acquire_release<T, MaybeVolatile>(store, load);
  }

  // memory_order::seq_cst
  {
    auto fetch_sub = [](MaybeVolatile<std::atomic<T>>& x, T old_value, T new_val) { x.fetch_sub(old_value - new_val); };
    auto fetch_sub_with_order = [](MaybeVolatile<std::atomic<T>>& x, T old_value, T new_val) {
      x.fetch_sub(old_value - new_val, std::memory_order::seq_cst);
    };
    auto load = [](auto& x) { return x.load(); };
    test_seq_cst<T, MaybeVolatile>(fetch_sub, load);
    test_seq_cst<T, MaybeVolatile>(fetch_sub_with_order, load);
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
