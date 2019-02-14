//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class piecewise_constant_distribution

// param_type param() const;

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::piecewise_constant_distribution<> D;
        typedef D::param_type P;
        double b[] = {10, 14, 16, 17};
        double p[] = {25, 62.5, 12.5};
        const size_t Np = sizeof(p) / sizeof(p[0]);
        P pa(b, b+Np+1, p);
        D d(pa);
        assert(d.param() == pa);
    }

  return 0;
}
