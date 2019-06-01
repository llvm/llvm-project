//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class lognormal_distribution

// explicit lognormal_distribution(result_type mean = 0, result_type stddev = 1);

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::lognormal_distribution<> D;
        D d;
        assert(d.m() == 0);
        assert(d.s() == 1);
    }
    {
        typedef std::lognormal_distribution<> D;
        D d(14.5);
        assert(d.m() == 14.5);
        assert(d.s() == 1);
    }
    {
        typedef std::lognormal_distribution<> D;
        D d(14.5, 5.25);
        assert(d.m() == 14.5);
        assert(d.s() == 5.25);
    }

  return 0;
}
