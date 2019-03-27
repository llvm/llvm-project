//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype;

// charT tolower(charT) const;

#include <locale>
#include <cassert>

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<wchar_t> F;
        const F& f = std::use_facet<F>(l);

        assert(f.tolower(L' ') == L' ');
        assert(f.tolower(L'A') == L'a');
        assert(f.tolower(L'\x07') == L'\x07');
        assert(f.tolower(L'.') == L'.');
        assert(f.tolower(L'a') == L'a');
        assert(f.tolower(L'1') == L'1');
    }

  return 0;
}
