//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept;

// Checks that __apply() can be called concurrently from multiple threads.

#include <__pstl/backends/statpart.h>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <thread>
#include <latch>
#include <vector>
#include "make_test_thread.h"

int main(int, char**) {
  constexpr std::size_t number_of_threads = 8;
  static constexpr std::size_t iters      = 32;
  std::vector<std::thread> threads(number_of_threads);
  std::latch started(static_cast<std::ptrdiff_t>(number_of_threads));
  std::atomic_size_t accum{0};

  // Start a number of threads
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = support::make_test_thread([&accum, &started] {
      // Wait until all have started
      started.arrive_and_wait();
      // Concurrently enqueue a new task from each of the threads
      std::__pstl::__statpart::__apply(iters, &accum, [](void* context1, std::size_t) {
        std::__pstl::__statpart::__apply(iters, context1, [](void* context2, std::size_t) {
          static_cast<std::atomic_size_t*>(context2)->fetch_add(1, std::memory_order_relaxed);
        });
      });
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  assert(accum.load() == number_of_threads * iters * iters);
  return 0;
}
