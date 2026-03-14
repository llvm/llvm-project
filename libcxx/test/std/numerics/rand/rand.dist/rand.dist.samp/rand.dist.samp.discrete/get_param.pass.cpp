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

// param_type param() const;

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::discrete_distribution<> D;
        typedef D::param_type P;
        double p0[] = {.3, .1, .6};
        P p(p0, p0+3);
        D d(p);
        assert(d.param() == p);
    }

  return 0;
}
