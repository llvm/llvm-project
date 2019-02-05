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

// explicit poisson_distribution(RealType lambda = 1.0);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::poisson_distribution<> D;
        D d;
        assert(d.mean() == 1);
    }
    {
        typedef std::poisson_distribution<> D;
        D d(3.5);
        assert(d.mean() == 3.5);
    }

  return 0;
}
