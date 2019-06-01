//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype;

// charT widen(char c) const;

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<wchar_t> F;
        const F& f = std::use_facet<F>(l);

        assert(f.widen(' ') == L' ');
        assert(f.widen('A') == L'A');
        assert(f.widen('\x07') == L'\x07');
        assert(f.widen('.') == L'.');
        assert(f.widen('a') == L'a');
        assert(f.widen('1') == L'1');
    }

  return 0;
}
