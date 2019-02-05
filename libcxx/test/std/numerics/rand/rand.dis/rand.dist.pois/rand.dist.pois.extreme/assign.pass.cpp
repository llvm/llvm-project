//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class extreme_value_distribution

// extreme_value_distribution& operator=(const extreme_value_distribution&);

#include <random>
#include <cassert>

void
test1()
{
    typedef std::extreme_value_distribution<> D;
    D d1(.5, 2);
    D d2;
    assert(d1 != d2);
    d2 = d1;
    assert(d1 == d2);
}

int main(int, char**)
{
    test1();

  return 0;
}
