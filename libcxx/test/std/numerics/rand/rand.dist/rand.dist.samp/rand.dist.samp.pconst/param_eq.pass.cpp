//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class piecewise_constant_distribution
// {
//     class param_type;

#include <random>
#include <limits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::piecewise_constant_distribution<> D;
        typedef D::param_type P;
        double b[] = {10, 14, 16, 17};
        double p[] = {25, 62.5, 12.5};
        P p1(b, b+4, p);
        P p2(b, b+4, p);
        assert(p1 == p2);
    }
    {
        typedef std::piecewise_constant_distribution<> D;
        typedef D::param_type P;
        double b[] = {10, 14, 16, 17};
        double p[] = {25, 62.5, 12.5};
        P p1(b, b+3, p);
        P p2(b, b+4, p);
        assert(p1 != p2);
    }

  return 0;
}
