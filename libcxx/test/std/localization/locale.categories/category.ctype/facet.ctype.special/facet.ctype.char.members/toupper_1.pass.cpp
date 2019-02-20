//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>;

// char toupper(char) const;

#include <locale>
#include <cassert>

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<char> F;
        const F& f = std::use_facet<F>(l);

        assert(f.toupper(' ') == ' ');
        assert(f.toupper('A') == 'A');
        assert(f.toupper('\x07') == '\x07');
        assert(f.toupper('.') == '.');
        assert(f.toupper('a') == 'A');
        assert(f.toupper('1') == '1');
    }

  return 0;
}
