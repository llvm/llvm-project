//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype;

// char narrow(charT c, char dfault) const;

#include <locale>
#include <cassert>

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<wchar_t> F;
        const F& f = std::use_facet<F>(l);

        assert(f.narrow(L' ', '*') == ' ');
        assert(f.narrow(L'A', '*') == 'A');
        assert(f.narrow(L'\x07', '*') == '\x07');
        assert(f.narrow(L'.', '*') == '.');
        assert(f.narrow(L'a', '*') == 'a');
        assert(f.narrow(L'1', '*') == '1');
    }

  return 0;
}
