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

// Checks that __apply() works with recursive calls up to at least 10 levels.

#include <__pstl/backends/statpart.h>
#include <atomic>
#include <cassert>
#include <cstddef>

template <int depth>
struct recurse {
  static void func(void* context, std::size_t iteration) {
    if (iteration == 1) {
      static_cast<std::atomic_size_t*>(context)->fetch_add(1, std::memory_order_relaxed);
      if constexpr (depth > 0) {
        std::__pstl::__statpart::__apply(2, context, &recurse<depth - 1>::func);
      }
    }
  }
};

int main(int, char**) {
  std::atomic_size_t accum{0};
  std::__pstl::__statpart::__apply(2, &accum, &recurse<9>::func);
  assert(accum.load() == 10);
  return 0;
}
