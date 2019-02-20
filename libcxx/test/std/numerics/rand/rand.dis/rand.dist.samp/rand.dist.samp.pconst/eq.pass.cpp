//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class piecewise_constant_distribution

// bool operator=(const piecewise_constant_distribution& x,
//                const piecewise_constant_distribution& y);
// bool operator!(const piecewise_constant_distribution& x,
//                const piecewise_constant_distribution& y);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::piecewise_constant_distribution<> D;
        D d1;
        D d2;
        assert(d1 == d2);
    }
    {
        typedef std::piecewise_constant_distribution<> D;
        double b[] = {10, 14, 16, 17};
        double p[] = {25, 62.5, 12.5};
        D d1(b, b+4, p);
        D d2(b, b+4, p);
        assert(d1 == d2);
    }
    {
        typedef std::piecewise_constant_distribution<> D;
        double b[] = {10, 14, 16, 17};
        double p[] = {25, 62.5, 12.5};
        D d1(b, b+4, p);
        D d2;
        assert(d1 != d2);
    }

  return 0;
}
