//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class chi_squared_distribution

// bool operator=(const chi_squared_distribution& x,
//                const chi_squared_distribution& y);
// bool operator!(const chi_squared_distribution& x,
//                const chi_squared_distribution& y);

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::chi_squared_distribution<> D;
        D d1(2.5);
        D d2(2.5);
        assert(d1 == d2);
    }
    {
        typedef std::chi_squared_distribution<> D;
        D d1(4);
        D d2(4.5);
        assert(d1 != d2);
    }

  return 0;
}
