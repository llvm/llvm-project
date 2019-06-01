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

// explicit exponential_distribution(RealType lambda = 1.0);

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::exponential_distribution<> D;
        D d;
        assert(d.lambda() == 1);
    }
    {
        typedef std::exponential_distribution<> D;
        D d(3.5);
        assert(d.lambda() == 3.5);
    }

  return 0;
}
