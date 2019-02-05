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

// explicit chi_squared_distribution(result_type alpha = 0, result_type beta = 1);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::chi_squared_distribution<> D;
        D d;
        assert(d.n() == 1);
    }
    {
        typedef std::chi_squared_distribution<> D;
        D d(14.5);
        assert(d.n() == 14.5);
    }

  return 0;
}
