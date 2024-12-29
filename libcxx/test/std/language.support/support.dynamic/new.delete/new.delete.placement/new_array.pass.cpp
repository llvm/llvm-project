//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test placement new array

#include <new>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

int A_constructed = 0;

struct A {
  A() { ++A_constructed; }
  ~A() { --A_constructed; }
};

TEST_CONSTEXPR_OPERATOR_NEW void test_direct_call() {
  assert(::operator new[](sizeof(int), &A_constructed) == &A_constructed);

  char ch = '*';
  assert(::operator new[](1, &ch) == &ch);
  assert(ch == '*');
}

#ifdef __cpp_lib_constexpr_new
static_assert((test_direct_call(), true));
#endif

int main(int, char**) {
  const std::size_t Size = 3;
  // placement new might require additional space.
  const std::size_t ExtraSize = 64;
  char buf[Size * sizeof(A) + ExtraSize];

  A* ap = new (buf) A[Size];
  assert((char*)ap >= buf);
  assert((char*)ap < (buf + ExtraSize));
  assert(A_constructed == Size);

  test_direct_call();
  return 0;
}
