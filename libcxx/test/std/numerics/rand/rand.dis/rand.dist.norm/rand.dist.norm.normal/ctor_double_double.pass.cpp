//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class normal_distribution

// explicit normal_distribution(result_type mean = 0, result_type stddev = 1);

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::normal_distribution<> D;
        D d;
        assert(d.mean() == 0);
        assert(d.stddev() == 1);
    }
    {
        typedef std::normal_distribution<> D;
        D d(14.5);
        assert(d.mean() == 14.5);
        assert(d.stddev() == 1);
    }
    {
        typedef std::normal_distribution<> D;
        D d(14.5, 5.25);
        assert(d.mean() == 14.5);
        assert(d.stddev() == 5.25);
    }

  return 0;
}
