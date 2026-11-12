//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio:  The static data members num and den shall have the common
//    divisor of the absolute values of N and D:

#include <ratio>

#include "test_macros.h"

template <long long N, long long D, long long eN, long long eD>
void test()
{
    static_assert((std::ratio<N, D>::num == eN), "");
    static_assert((std::ratio<N, D>::den == eD), "");
}

int main(int, char**)
{
    test<1, 1, 1, 1>();
    test<1, 10, 1, 10>();
    test<10, 10, 1, 1>();
    test<10, 1, 10, 1>();
    test<12, 4, 3, 1>();
    test<12, -4, -3, 1>();
    test<-12, 4, -3, 1>();
    test<-12, -4, 3, 1>();
    test<4, 12, 1, 3>();
    test<4, -12, -1, 3>();
    test<-4, 12, -1, 3>();
    test<-4, -12, 1, 3>();
    test<222, 333, 2, 3>();
    test<222, -333, -2, 3>();
    test<-222, 333, -2, 3>();
    test<-222, -333, 2, 3>();
    test<0x7FFFFFFFFFFFFFFFLL, 127, 72624976668147841LL, 1>();
    test<-0x7FFFFFFFFFFFFFFFLL, 127, -72624976668147841LL, 1>();
    test<0x7FFFFFFFFFFFFFFFLL, -127, -72624976668147841LL, 1>();
    test<-0x7FFFFFFFFFFFFFFFLL, -127, 72624976668147841LL, 1>();

  return 0;
}
