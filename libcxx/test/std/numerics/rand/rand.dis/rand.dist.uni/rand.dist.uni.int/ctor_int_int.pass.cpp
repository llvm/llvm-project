//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class _IntType = int>
// class uniform_int_distribution

// explicit uniform_int_distribution(IntType a = 0,
//                                   IntType b = numeric_limits<IntType>::max());

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::uniform_int_distribution<> D;
        D d;
        assert(d.a() == 0);
        assert(d.b() == std::numeric_limits<int>::max());
    }
    {
        typedef std::uniform_int_distribution<> D;
        D d(-6);
        assert(d.a() == -6);
        assert(d.b() == std::numeric_limits<int>::max());
    }
    {
        typedef std::uniform_int_distribution<> D;
        D d(-6, 106);
        assert(d.a() == -6);
        assert(d.b() == 106);
    }

  return 0;
}
