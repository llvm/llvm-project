//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-localization

// <complex>

// template<class T, class charT, class traits>
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& o, const complex<T>& x);

#include <complex>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
    // Basic test
    {
        std::complex<double> const c(1, 2);
        std::ostringstream os;
        os << c;
        assert(os.str() == "(1,2)");
    }

    // Test with various widths.
    // In particular, make sure the width() is 0 after the operation, which
    // should be the case because [complex.ops] says about operator<< for complex:
    //
    //   Effects: Inserts the complex number x onto the stream o as if it
    //   were implemented as follows:
    //
    //     basic_ostringstream<charT, traits> s;
    //     s.flags(o.flags());
    //     s.imbue(o.getloc());
    //     s.precision(o.precision());
    //     s << '(' << x.real() << "," << x.imag() << ')';
    //     return o << s.str();
    //
    // Since operator<< for std::string sets o.width(0), operator<< for
    // std::complex should do the same.
    {
        for (int width = 0; width <= 5; ++width) {
            std::complex<double> const c(1, 2);
            std::ostringstream os;
            os.width(width);
            os.fill('_');
            os << c;
            assert(os.width() == 0);
            assert(os.str() == "(1,2)");
        }
        {
            std::complex<double> const c(1, 2);
            std::ostringstream os;
            os.width(6);
            os.fill('_');
            os << c;
            assert(os.width() == 0);
            assert(os.str() == "_(1,2)");
        }
        {
            std::complex<double> const c(1, 2);
            std::ostringstream os;
            os.width(7);
            os.fill('_');
            os << c;
            assert(os.width() == 0);
            assert(os.str() == "__(1,2)");
        }
        {
            std::complex<double> const c(1, 2);
            std::ostringstream os;
            os.width(8);
            os.fill('_');
            os << c;
            assert(os.width() == 0);
            assert(os.str() == "___(1,2)");
        }
        // Insert something after the complex and make sure the
        // stream's width has been reset as expected.
        {
            std::complex<double> const c(1, 2);
            std::ostringstream os;
            os.width(8);
            os.fill('_');
            os << c;
            assert(os.width() == 0);

            os << "hello";
            assert(os.str() == "___(1,2)hello");
        }

        // Test with numbers that result in different output lengths, to
        // make sure we handle custom width() correctly.
        {
            std::complex<double> const c(123, 456);
            std::ostringstream os;
            os.width(4);
            os.fill('_');
            os << c;
            assert(os.width() == 0);
            assert(os.str() == "(123,456)");
        }
        {
            std::complex<double> const c(123, 456);
            std::ostringstream os;
            os.width(12);
            os.fill('_');
            os << c;
            assert(os.width() == 0);
            assert(os.str() == "___(123,456)");

            os << "hello";
            assert(os.str() == "___(123,456)hello");
        }

        // Make sure left fill behaves correctly
        {
            std::complex<double> const c(123, 456);
            std::ostringstream os;
            os.width(12);
            os.fill('_');
            os << std::left << c;
            assert(os.width() == 0);
            assert(os.str() == "(123,456)___");

            os << "xy";
            assert(os.str() == "(123,456)___xy");
        }
    }

    return 0;
}
