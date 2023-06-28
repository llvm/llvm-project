//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// floating-point-type fetch_add(floating-point-type,
//                               memory_order = memory_order::seq_cst) volatile noexcept;
// floating-point-type fetch_add(floating-point-type,
//                               memory_order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <thread>
#include <vector>

#include "test_macros.h"
#include "make_test_thread.h"

template <class T>
void test() {
  // fetch_add
  {
    std::atomic<T> a(3.1);
    std::same_as<T> decltype(auto) r = a.fetch_add(1.2);
    assert(r == T(3.1));
    assert(a.load() == T(3.1) + T(1.2));
  }

  // fetch_add volatile
  {
    volatile std::atomic<T> a(3.1);
    std::same_as<T> decltype(auto) r = a.fetch_add(1.2);
    assert(r == T(3.1));
    assert(a.load() == T(3.1) + T(1.2));
  }

  // fetch_add concurrent
  {
    constexpr auto numberOfThreads = 4;
    constexpr auto loop            = 1000;

    std::atomic<T> at;

    std::vector<std::thread> threads;
    threads.reserve(numberOfThreads);
    for (auto i = 0; i < numberOfThreads; ++i) {
      threads.emplace_back([&at]() {
        for (auto j = 0; j < loop; ++j) {
          at.fetch_add(T(1.234));
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

    assert(at.load() == times(1.234, numberOfThreads * loop));
  }
}

int main(int, char**) {
  test<float>();
  test<double>();
  test<long double>();

  return 0;
}
