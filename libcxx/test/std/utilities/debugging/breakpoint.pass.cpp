//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <debugging>

// void breakpoint() noexcept;

#include <cassert>
#include <debugging>

void test() { static_assert(noexcept(breakpoint())); }

int main(int, char**) {
  test();

  return 0;
}
