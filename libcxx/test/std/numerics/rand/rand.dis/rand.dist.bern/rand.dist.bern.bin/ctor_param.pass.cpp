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

// explicit binomial_distribution(const param_type& parm);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::binomial_distribution<> D;
        typedef D::param_type P;
        P p(5, 0.25);
        D d(p);
        assert(d.t() == 5);
        assert(d.p() == 0.25);
    }

  return 0;
}
