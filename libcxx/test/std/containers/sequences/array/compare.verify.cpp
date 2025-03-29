//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// template <class T, size_t N>
//   bool operator==(const array<T,N>& x, const array<T,N>& y);    // constexpr in C++20
// template <class T, size_t N>
//   bool operator!=(const array<T,N>& x, const array<T,N>& y);    // removed in C++20
// template <class T, size_t N>
//   bool operator<(const array<T,N>& x, const array<T,N>& y);     // removed in C++20
// template <class T, size_t N>
//   bool operator>(const array<T,N>& x, const array<T,N>& y);     // removed in C++20
// template <class T, size_t N>
//   bool operator<=(const array<T,N>& x, const array<T,N>& y);    // removed in C++20
// template <class T, size_t N>
//   bool operator>=(const array<T,N>& x, const array<T,N>& y);    // removed in C++20

#include <array>

#include "test_macros.h"

template <int>
struct NoCompare {};

int main(int, char**) {
  {
    typedef NoCompare<0> T;
    typedef std::array<T, 3> C;
    C c1 = {{}};
    // expected-error@*:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD(c1 == c1);
    TEST_IGNORE_NODISCARD(c1 < c1);
  }
  {
    typedef NoCompare<1> T;
    typedef std::array<T, 3> C;
    C c1 = {{}};
    // expected-error@*:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD(c1 != c1);
    TEST_IGNORE_NODISCARD(c1 > c1);
  }
  {
    typedef NoCompare<2> T;
    typedef std::array<T, 0> C;
    C c1 = {{}};
    // expected-error@*:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD(c1 == c1);
    TEST_IGNORE_NODISCARD(c1 < c1);
  }

  return 0;
}
