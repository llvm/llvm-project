//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept;

// Checks that __apply() supports fanning out on 3 levels.

#include <__pstl/backends/statpart.h>
#include <atomic>
#include <cassert>
#include <cstddef>

int main(int, char**) {
  for (int round = 0; round < 10; ++round) {
    static constexpr size_t iters = 32;
    std::atomic_size_t accum{0};
    std::__pstl::__statpart::__apply(iters, &accum, [](void* context1, std::size_t) {
      std::__pstl::__statpart::__apply(iters, context1, [](void* context2, std::size_t) {
        std::__pstl::__statpart::__apply(iters, context2, [](void* context3, std::size_t) {
          static_cast<std::atomic_size_t*>(context3)->fetch_add(1, std::memory_order_relaxed);
        });
      });
    });
    assert(accum.load() == iters * iters * iters);
  }
  return 0;
}
