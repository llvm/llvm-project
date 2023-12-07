//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>

// static const mask* classic_table() throw();

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::ctype<char> F;
    assert(F::classic_table() != 0);
    assert(F::table_size >= 256);

    typedef F::mask mask;
    const mask *p = F::classic_table();

    for ( std::size_t i = 0; i < 128; ++i ) // values above 128 are not consistent
    {

        bool expect_cntrl = (i < 32 || 126 < i);
        bool expect_print = (32 <= i && i <= 126);

        bool expect_space = (9 <= i && i <= 13) || (i == ' ');
#if defined(_MSVC_STL_VERSION)
        // MS STL includes the _SPACE bit in F::blank
        bool expect_blank = (9 <= i && i <= 13) || (i == ' ');
#elif defined(_WIN32)
        // The _BLANK bit isn't set for '\t' on Windows
        bool expect_blank = (i == ' ');
#else
        bool expect_blank = (i == '\t' || i == ' ');
#endif

        bool expect_upper = ('A' <= i && i <= 'Z');
        bool expect_lower = ('a' <= i && i <= 'z');
        bool expect_alpha = expect_upper || expect_lower;

        bool expect_digit = ('0' <= i && i <= '9');
        bool expect_xdigit = ('A' <= i && i <= 'F')
                          || ('a' <= i && i <= 'f')
                          || expect_digit;

        bool expect_punct = (33 <= i && i <= 47)     // ' ' .. '/'
                         || (58 <= i && i <= 64)     // ':' .. '@'
                         || (91 <= i && i <= 96)     // '[' .. '`'
                         || (123 <= i && i <= 126);  // '{' .. '~'

        assert(bool(p[i] & F::cntrl) == expect_cntrl);
        assert(bool(p[i] & F::print) == expect_print);
        assert(bool(p[i] & F::space) == expect_space);
        assert(bool(p[i] & F::blank) == expect_blank);
        assert(bool(p[i] & F::lower) == expect_lower);
        assert(bool(p[i] & F::upper) == expect_upper);
        assert(bool(p[i] & F::alpha) == expect_alpha);
        assert(bool(p[i] & F::digit) == expect_digit);
        assert(bool(p[i] & F::xdigit) == expect_xdigit);
        assert(bool(p[i] & F::punct) == expect_punct);
    }


  return 0;
}
