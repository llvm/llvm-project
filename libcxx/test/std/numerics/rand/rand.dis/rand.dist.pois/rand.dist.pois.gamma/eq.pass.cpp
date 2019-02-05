//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class gamma_distribution

// bool operator=(const gamma_distribution& x,
//                const gamma_distribution& y);
// bool operator!(const gamma_distribution& x,
//                const gamma_distribution& y);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::gamma_distribution<> D;
        D d1(2.5, 4);
        D d2(2.5, 4);
        assert(d1 == d2);
    }
    {
        typedef std::gamma_distribution<> D;
        D d1(2.5, 4);
        D d2(2.5, 4.5);
        assert(d1 != d2);
    }

  return 0;
}
