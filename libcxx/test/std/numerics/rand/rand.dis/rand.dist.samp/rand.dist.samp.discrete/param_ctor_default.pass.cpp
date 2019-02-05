//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <random>

// template<class IntType = int>
// class discrete_distribution

// param_type(initializer_list<double> wl);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::discrete_distribution<> D;
        typedef D::param_type P;
        P pa = {1};
        std::vector<double> p = pa.probabilities();
        assert(p.size() == 1);
        assert(p[0] == 1);
    }

  return 0;
}
