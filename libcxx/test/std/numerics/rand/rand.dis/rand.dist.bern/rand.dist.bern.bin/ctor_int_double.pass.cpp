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

// explicit binomial_distribution(IntType t = 1, double p = 0.5);

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::binomial_distribution<> D;
        D d;
        assert(d.t() == 1);
        assert(d.p() == 0.5);
    }
    {
        typedef std::binomial_distribution<> D;
        D d(3);
        assert(d.t() == 3);
        assert(d.p() == 0.5);
    }
    {
        typedef std::binomial_distribution<> D;
        D d(3, 0.75);
        assert(d.t() == 3);
        assert(d.p() == 0.75);
    }

  return 0;
}
