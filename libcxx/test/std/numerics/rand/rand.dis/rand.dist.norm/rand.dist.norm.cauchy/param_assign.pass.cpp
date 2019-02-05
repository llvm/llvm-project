//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class cauchy_distribution
// {
//     class param_type;

#include <random>
#include <limits>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::cauchy_distribution<> D;
        typedef D::param_type param_type;
        param_type p0(.75, 6);
        param_type p;
        p = p0;
        assert(p.a() == .75);
        assert(p.b() == 6);
    }

  return 0;
}
