//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class discrete_distribution

// bool operator=(const discrete_distribution& x,
//                const discrete_distribution& y);
// bool operator!(const discrete_distribution& x,
//                const discrete_distribution& y);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::discrete_distribution<> D;
        D d1;
        D d2;
        assert(d1 == d2);
    }
    {
        typedef std::discrete_distribution<> D;
        double p0[] = {1};
        D d1(p0, p0+1);
        D d2;
        assert(d1 == d2);
    }
    {
        typedef std::discrete_distribution<> D;
        double p0[] = {10, 30};
        D d1(p0, p0+2);
        D d2;
        assert(d1 != d2);
    }
}
