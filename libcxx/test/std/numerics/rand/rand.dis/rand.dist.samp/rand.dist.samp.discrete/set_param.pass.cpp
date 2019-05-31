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

// void param(const param_type& parm);

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::discrete_distribution<> D;
        typedef D::param_type P;
        double d0[] = {.3, .1, .6};
        P p(d0, d0+3);
        D d;
        d.param(p);
        assert(d.param() == p);
    }

  return 0;
}
