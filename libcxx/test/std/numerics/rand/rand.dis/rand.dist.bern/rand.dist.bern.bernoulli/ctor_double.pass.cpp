//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// explicit bernoulli_distribution(double p = 0.5);

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::bernoulli_distribution D;
        D d;
        assert(d.p() == 0.5);
    }
    {
        typedef std::bernoulli_distribution D;
        D d(0);
        assert(d.p() == 0);
    }
    {
        typedef std::bernoulli_distribution D;
        D d(0.75);
        assert(d.p() == 0.75);
    }

  return 0;
}
