//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class poisson_distribution

// bool operator=(const poisson_distribution& x,
//                const poisson_distribution& y);
// bool operator!(const poisson_distribution& x,
//                const poisson_distribution& y);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::poisson_distribution<> D;
        D d1(.25);
        D d2(.25);
        assert(d1 == d2);
    }
    {
        typedef std::poisson_distribution<> D;
        D d1(.28);
        D d2(.25);
        assert(d1 != d2);
    }

  return 0;
}
