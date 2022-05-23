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

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::binomial_distribution<> D;
        typedef D::param_type param_type;
        param_type p1(3, 0.75);
        param_type p2(3, 0.75);
        assert(p1 == p2);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef D::param_type param_type;
        param_type p1(3, 0.75);
        param_type p2(3, 0.5);
        assert(p1 != p2);
    }

  return 0;
}
