//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class uniform_real_distribution
// {
//     class param_type;

#include <random>
#include <limits>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::uniform_real_distribution<float> D;
        typedef D::param_type param_type;
        param_type p1(5, 10);
        param_type p2(5, 10);
        assert(p1 == p2);
    }
    {
        typedef std::uniform_real_distribution<float> D;
        typedef D::param_type param_type;
        param_type p1(5, 10);
        param_type p2(6, 10);
        assert(p1 != p2);
    }

  return 0;
}
