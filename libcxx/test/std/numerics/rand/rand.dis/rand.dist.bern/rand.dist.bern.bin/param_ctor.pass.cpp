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
// {
//     class param_type;

#include <random>
#include <limits>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::binomial_distribution<> D;
        typedef D::param_type param_type;
        param_type p;
        assert(p.t() == 1);
        assert(p.p() == 0.5);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef D::param_type param_type;
        param_type p(10);
        assert(p.t() == 10);
        assert(p.p() == 0.5);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef D::param_type param_type;
        param_type p(10, 0.25);
        assert(p.t() == 10);
        assert(p.p() == 0.25);
    }

  return 0;
}
