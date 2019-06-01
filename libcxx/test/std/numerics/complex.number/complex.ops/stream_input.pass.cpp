//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T, class charT, class traits>
//   basic_istream<charT, traits>&
//   operator>>(basic_istream<charT, traits>& is, complex<T>& x);

#include <complex>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::istringstream is("5");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(5, 0));
        assert(is.eof());
    }
    {
        std::istringstream is(" 5 ");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(5, 0));
        assert(is.good());
    }
    {
        std::istringstream is(" 5, ");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(5, 0));
        assert(is.good());
    }
    {
        std::istringstream is(" , 5, ");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(0, 0));
        assert(is.fail());
    }
    {
        std::istringstream is("5.5 ");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(5.5, 0));
        assert(is.good());
    }
    {
        std::istringstream is(" ( 5.5 ) ");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(5.5, 0));
        assert(is.good());
    }
    {
        std::istringstream is("  5.5)");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(5.5, 0));
        assert(is.good());
    }
    {
        std::istringstream is("(5.5 ");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(0, 0));
        assert(is.fail());
    }
    {
        std::istringstream is("(5.5,");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(0, 0));
        assert(is.fail());
    }
    {
        std::istringstream is("( -5.5 , -6.5 )");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(-5.5, -6.5));
        assert(!is.eof());
    }
    {
        std::istringstream is("(-5.5,-6.5)");
        std::complex<double> c;
        is >> c;
        assert(c == std::complex<double>(-5.5, -6.5));
        assert(!is.eof());
    }

  return 0;
}
