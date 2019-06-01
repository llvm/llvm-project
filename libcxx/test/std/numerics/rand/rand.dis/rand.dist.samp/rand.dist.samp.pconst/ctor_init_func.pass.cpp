//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <random>

// template<class RealType = double>
// class piecewise_constant_distribution

// piecewise_constant_distribution(initializer_list<result_type> bl,
//                                 UnaryOperation fw);

#include <random>
#include <cassert>

#include "test_macros.h"

double f(double x)
{
    return x*2;
}

int main(int, char**)
{
    {
        typedef std::piecewise_constant_distribution<> D;
        D d({}, f);
        std::vector<double> iv = d.intervals();
        assert(iv.size() == 2);
        assert(iv[0] == 0);
        assert(iv[1] == 1);
        std::vector<double> dn = d.densities();
        assert(dn.size() == 1);
        assert(dn[0] == 1);
    }
    {
        typedef std::piecewise_constant_distribution<> D;
        D d({12}, f);
        std::vector<double> iv = d.intervals();
        assert(iv.size() == 2);
        assert(iv[0] == 0);
        assert(iv[1] == 1);
        std::vector<double> dn = d.densities();
        assert(dn.size() == 1);
        assert(dn[0] == 1);
    }
    {
        typedef std::piecewise_constant_distribution<> D;
        D d({12, 14}, f);
        std::vector<double> iv = d.intervals();
        assert(iv.size() == 2);
        assert(iv[0] == 12);
        assert(iv[1] == 14);
        std::vector<double> dn = d.densities();
        assert(dn.size() == 1);
        assert(dn[0] == 0.5);
    }
    {
        typedef std::piecewise_constant_distribution<> D;
        D d({5.5, 7.5, 11.5}, f);
        std::vector<double> iv = d.intervals();
        assert(iv.size() == 3);
        assert(iv[0] == 5.5);
        assert(iv[1] == 7.5);
        assert(iv[2] == 11.5);
        std::vector<double> dn = d.densities();
        assert(dn.size() == 2);
        assert(dn[0] == 0.203125);
        assert(dn[1] == 0.1484375);
    }

  return 0;
}
