//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++17, libcpp-pstl-backend-std-thread

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// void __apply(size_t __iterations, void* __context, void (*__func)(void* __context, size_t __iteration)) noexcept;

#include <__pstl/backends/std_thread.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>
#include <cstdio>

// Fork-bomb Fibonacci implementation.
int fibb(int n) {
  if (n <= 1)
    return n;

  struct Ctx {
    int n;
    int n12[2];
  } ctx{.n = n, .n12 = {0, 0}};
  std::__pstl::__std_thread::__apply(2, &ctx, [](void* ctxt, std::size_t i) {
    auto& c  = *static_cast<Ctx*>(ctxt);
    c.n12[i] = fibb(c.n - 1 - i);
  });

  return ctx.n12[0] + ctx.n12[1];
}

// Flat fork-join style application.
bool flat_fork_join() {
  std::vector<int> v(1'000'000, 0);
  std::__pstl::__std_thread::__apply(v.size(), v.data(), [](void* data, std::size_t i) {
    static_cast<int*>(data)[i] = 42;
  });
  return std::all_of(v.begin(), v.end(), [](int x) { return x == 42; });
}

int main(int, char**) {
  assert(fibb(20) == 6765);
  // assert(fibb(25) == 75025);
  assert(flat_fork_join());
  return 0;
}
