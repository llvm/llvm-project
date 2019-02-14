//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class chi_squared_distribution
// {
//     class param_type;

#include <random>
#include <limits>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::chi_squared_distribution<> D;
        typedef D::param_type param_type;
        param_type p;
        assert(p.n() == 1);
    }
    {
        typedef std::chi_squared_distribution<> D;
        typedef D::param_type param_type;
        param_type p(10);
        assert(p.n() == 10);
    }

  return 0;
}
