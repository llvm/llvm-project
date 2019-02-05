//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class gamma_distribution
// {
//     class param_type;

#include <random>
#include <limits>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::gamma_distribution<> D;
        typedef D::param_type param_type;
        param_type p;
        assert(p.alpha() == 1);
        assert(p.beta() == 1);
    }
    {
        typedef std::gamma_distribution<> D;
        typedef D::param_type param_type;
        param_type p(10);
        assert(p.alpha() == 10);
        assert(p.beta() == 1);
    }
    {
        typedef std::gamma_distribution<> D;
        typedef D::param_type param_type;
        param_type p(10, 5);
        assert(p.alpha() == 10);
        assert(p.beta() == 5);
    }

  return 0;
}
