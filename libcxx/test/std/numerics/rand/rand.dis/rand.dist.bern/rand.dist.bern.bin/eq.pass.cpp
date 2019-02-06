//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class binomial_distribution

// bool operator=(const binomial_distribution& x,
//                const binomial_distribution& y);
// bool operator!(const binomial_distribution& x,
//                const binomial_distribution& y);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::binomial_distribution<> D;
        D d1(3, .25);
        D d2(3, .25);
        assert(d1 == d2);
    }
    {
        typedef std::binomial_distribution<> D;
        D d1(3, .28);
        D d2(3, .25);
        assert(d1 != d2);
    }
    {
        typedef std::binomial_distribution<> D;
        D d1(3, .25);
        D d2(4, .25);
        assert(d1 != d2);
    }
}
