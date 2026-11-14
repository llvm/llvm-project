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

// Checks that __apply() can be invoked after the main() function finishes.

#include <__pstl/backends/statpart.h>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>

int main(int, char**) {
  return std::atexit(+[] {
    static constexpr std::size_t iters = 32;
    std::atomic_size_t accum{0};
    std::__pstl::__statpart::__apply(iters, &accum, [](void* context, std::size_t) {
      static_cast<std::atomic_size_t*>(context)->fetch_add(1, std::memory_order_relaxed);
    });
    assert(accum.load() == iters);
  });
}
