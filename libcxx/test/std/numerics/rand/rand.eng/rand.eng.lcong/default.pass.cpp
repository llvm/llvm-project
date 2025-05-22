//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template <class UIntType, UIntType a, UIntType c, UIntType m>
//   class linear_congruential_engine;

// linear_congruential_engine();

#include <random>
#include <cassert>

#include "test_macros.h"

template <class T, T a, T c, T m>
void
test1()
{
    typedef std::linear_congruential_engine<T, a, c, m> LCE;
    LCE e1;
    LCE e2;
    e2.seed();
    assert(e1 == e2);
}

template <class T>
void
test()
{
  const int W = sizeof(T) * CHAR_BIT;
  const T M(static_cast<T>(-1));
  const T A(static_cast<T>((static_cast<T>(1) << (W / 2)) - 1));

  // Cases where m = 0
  test1<T, 0, 0, 0>();
  test1<T, A, 0, 0>();
  test1<T, 0, 1, 0>();
  test1<T, A, 1, 0>();

  // Cases where m = 2^n for n < w
  test1<T, 0, 0, 256>();
  test1<T, 5, 0, 256>();
  test1<T, 0, 1, 256>();
  test1<T, 5, 1, 256>();

  // Cases where m is odd and a = 0
  test1<T, 0, 0, M>();
  test1<T, 0, M - 2, M>();
  test1<T, 0, M - 1, M>();

  // Cases where m is odd and m % a <= m / a (Schrage)
  test1<T, A, 0, M>();
  test1<T, A, M - 2, M>();
  test1<T, A, M - 1, M>();

  /*
  // Cases where m is odd and m % a > m / a (not implemented)
  test1<T, M - 2, 0, M>();
  test1<T, M - 2, M - 2, M>();
  test1<T, M - 2, M - 1, M>();
  test1<T, M - 1, 0, M>();
  test1<T, M - 1, M - 2, M>();
  test1<T, M - 1, M - 1, M>();
  */
}

int main(int, char**)
{
    test<unsigned short>();
    test<unsigned int>();
    test<unsigned long>();
    test<unsigned long long>();

  return 0;
}
