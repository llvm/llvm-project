//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// [valarray.syn]/3
//   Any function returning a valarray<T> is permitted to return an object of
//   another type, provided all the const member functions of valarray<T> are
//   also applicable to this type.
//
// Libc++ uses this and returns __val_expr<_Expr> for several operations.
//
// The const overloads of
//   valarray::operator[](...) const
// return proxy objects. These proxies are implicitly convertible to
// std::valarray.
//
// Validate the function works for valarray, the proxies, and __val_expr.
//
// valarray& operator<<=(const valarray& v);

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

template <class A>
void test(const A& rhs) {
  int input[]      = {1, 2, 3, 4, 5};
  int expected[]   = {64, 256, 768, 2048, 5120};
  const unsigned N = sizeof(input) / sizeof(input[0]);
  std::valarray<int> value(input, N);

  value <<= rhs;

  assert(value.size() == N);
  for (std::size_t i = 0; i < value.size(); ++i)
    assert(value[i] == expected[i]);
}

int main(int, char**) {
  int input[]      = {6, 7, 8, 9, 10};
  const unsigned N = sizeof(input) / sizeof(input[0]);

  std::valarray<bool> mask(true, N);
  std::size_t indices[] = {0, 1, 2, 3, 4};
  std::valarray<std::size_t> indirect(indices, N);

  std::valarray<int> zero(0, N);

  {
    std::valarray<int> value(input, N);

    test(value);
    test(value[std::slice(0, N, 1)]);
    test(value[std::gslice(0, std::valarray<std::size_t>(N, 1), std::valarray<std::size_t>(1, 1))]);
    test(value[mask]);
    test(value[indirect]);
    test(value + zero);
  }

  {
    const std::valarray<int> value(input, N);

    test(value);
    test(value[std::slice(0, N, 1)]);
    test(value[std::gslice(0, std::valarray<std::size_t>(N, 1), std::valarray<std::size_t>(1, 1))]);
    test(value[mask]);
    test(value[indirect]);
    test(value + zero);
  }

  return 0;
}
