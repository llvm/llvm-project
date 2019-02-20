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
// {
//     class param_type;

#include <random>
#include <limits>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::discrete_distribution<> D;
        typedef D::param_type param_type;
        double p0[] = {30, 10};
        param_type p1(p0, p0+2);
        param_type p2(p0, p0+2);
        assert(p1 == p2);
    }
    {
        typedef std::discrete_distribution<> D;
        typedef D::param_type param_type;
        double p0[] = {30, 10};
        param_type p1(p0, p0+2);
        param_type p2;
        assert(p1 != p2);
    }

  return 0;
}
