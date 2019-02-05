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

// explicit uniform_real_distribution(RealType a = 0,
//                                    RealType b = 1);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::uniform_real_distribution<> D;
        D d;
        assert(d.a() == 0);
        assert(d.b() == 1);
    }
    {
        typedef std::uniform_real_distribution<> D;
        D d(-6);
        assert(d.a() == -6);
        assert(d.b() == 1);
    }
    {
        typedef std::uniform_real_distribution<> D;
        D d(-6, 106);
        assert(d.a() == -6);
        assert(d.b() == 106);
    }

  return 0;
}
