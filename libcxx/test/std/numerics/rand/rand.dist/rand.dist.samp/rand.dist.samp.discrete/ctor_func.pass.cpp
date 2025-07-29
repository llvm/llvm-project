//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class discrete_distribution

// template<class UnaryOperation>
//     discrete_distribution(size_t nw, double xmin, double xmax,
//                           UnaryOperation fw);

// There is a bogus diagnostic about a too large allocation
// ADDITIONAL_COMPILE_FLAGS(gcc): -Wno-alloc-size-larger-than

#include <random>

#include <cassert>
#include <vector>

#include "test_macros.h"

double fw(double x)
{
    return x+1;
}

int main(int, char**)
{
    {
        typedef std::discrete_distribution<> D;
        D d(0, 0, 1, fw);
        std::vector<double> p = d.probabilities();
        assert(p.size() == 1);
        assert(p[0] == 1);
    }
    {
        typedef std::discrete_distribution<> D;
        D d(1, 0, 1, fw);
        std::vector<double> p = d.probabilities();
        assert(p.size() == 1);
        assert(p[0] == 1);
    }
    {
        typedef std::discrete_distribution<> D;
        D d(2, 0.5, 1.5, fw);
        std::vector<double> p = d.probabilities();
        assert(p.size() == 2);
        assert(p[0] == .4375);
        assert(p[1] == .5625);
    }
    {
        typedef std::discrete_distribution<> D;
        D d(4, 0, 2, fw);
        std::vector<double> p = d.probabilities();
        assert(p.size() == 4);
        assert(p[0] == .15625);
        assert(p[1] == .21875);
        assert(p[2] == .28125);
    }

  return 0;
}
