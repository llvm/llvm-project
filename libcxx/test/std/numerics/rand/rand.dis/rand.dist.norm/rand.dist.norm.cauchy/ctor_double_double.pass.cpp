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

// explicit cauchy_distribution(result_type a = 0, result_type b = 1);

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::cauchy_distribution<> D;
        D d;
        assert(d.a() == 0);
        assert(d.b() == 1);
    }
    {
        typedef std::cauchy_distribution<> D;
        D d(14.5);
        assert(d.a() == 14.5);
        assert(d.b() == 1);
    }
    {
        typedef std::cauchy_distribution<> D;
        D d(14.5, 5.25);
        assert(d.a() == 14.5);
        assert(d.b() == 5.25);
    }

  return 0;
}
