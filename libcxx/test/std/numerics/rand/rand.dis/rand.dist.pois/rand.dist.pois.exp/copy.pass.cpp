//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class exponential_distribution

// exponential_distribution(const exponential_distribution&);

#include <random>
#include <cassert>

void
test1()
{
    typedef std::exponential_distribution<> D;
    D d1(1.75);
    D d2 = d1;
    assert(d1 == d2);
}

int main()
{
    test1();
}
