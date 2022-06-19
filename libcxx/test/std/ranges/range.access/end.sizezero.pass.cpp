//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: msvc

// std::ranges::end
// std::ranges::cend
//   Test the fix for https://llvm.org/PR54100

#include <ranges>
#include <cassert>

#include "test_macros.h"

struct A {
  int m[0];
};
static_assert(sizeof(A) == 0); // an extension supported by GCC and Clang

int main(int, char**)
{
  A a[10];
  std::same_as<A*> auto p = std::ranges::end(a);
  assert(p == a + 10);
  std::same_as<const A*> auto cp = std::ranges::cend(a);
  assert(cp == a + 10);

  return 0;
}
