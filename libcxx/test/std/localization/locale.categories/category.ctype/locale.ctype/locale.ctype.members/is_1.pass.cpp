//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype;

// bool is(mask m, charT c) const;

// XFAIL: libcpp-has-no-wide-characters

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<wchar_t> F;
        const F& f = std::use_facet<F>(l);

        assert(f.is(F::space, L' '));
        assert(!f.is(F::space, L'A'));

        assert(f.is(F::print, L' '));
        assert(!f.is(F::print, L'\x07'));

        assert(f.is(F::cntrl, L'\x07'));
        assert(!f.is(F::cntrl, L' '));

        assert(f.is(F::upper, L'A'));
        assert(!f.is(F::upper, L'a'));

        assert(f.is(F::lower, L'a'));
        assert(!f.is(F::lower, L'A'));

        assert(f.is(F::alpha, L'a'));
        assert(!f.is(F::alpha, L'1'));

        assert(f.is(F::digit, L'1'));
        assert(!f.is(F::digit, L'a'));

        assert(f.is(F::punct, L'.'));
        assert(!f.is(F::punct, L'a'));

        assert(f.is(F::xdigit, L'a'));
        assert(!f.is(F::xdigit, L'g'));

        assert(f.is(F::alnum, L'a'));
        assert(!f.is(F::alnum, L'.'));

        assert(f.is(F::graph, L'.'));
        assert(!f.is(F::graph,  L'\x07'));
    }

  return 0;
}
