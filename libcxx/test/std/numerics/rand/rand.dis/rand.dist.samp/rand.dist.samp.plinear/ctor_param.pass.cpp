//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class piecewise_linear_distribution

// explicit piecewise_linear_distribution(const param_type& parm);

#include <random>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::piecewise_linear_distribution<> D;
        typedef D::param_type P;
        double b[] = {10, 14, 16, 17};
        double p[] = {25, 62.5, 12.5, 0};
        P pa(b, b+4, p);
        D d(pa);
        std::vector<double> iv = d.intervals();
        assert(iv.size() == 4);
        assert(iv[0] == 10);
        assert(iv[1] == 14);
        assert(iv[2] == 16);
        assert(iv[3] == 17);
        std::vector<double> dn = d.densities();
        assert(dn.size() == 4);
        assert(dn[0] == 25/256.25);
        assert(dn[1] == 62.5/256.25);
        assert(dn[2] == 12.5/256.25);
        assert(dn[3] == 0);
    }

  return 0;
}
