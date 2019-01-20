//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>;

// char tolower(char) const;

#include <locale>
#include <cassert>

int main()
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<char> F;
        const F& f = std::use_facet<F>(l);

        assert(f.tolower(' ') == ' ');
        assert(f.tolower('A') == 'a');
        assert(f.tolower('\x07') == '\x07');
        assert(f.tolower('.') == '.');
        assert(f.tolower('a') == 'a');
        assert(f.tolower('1') == '1');
    }
}
