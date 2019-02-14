//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>;

// char narrow(char c, char dfault) const;

#include <locale>
#include <cassert>

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<char> F;
        const F& f = std::use_facet<F>(l);

        assert(f.narrow(' ', '*') == ' ');
        assert(f.narrow('A', '*') == 'A');
        assert(f.narrow('\x07', '*') == '\x07');
        assert(f.narrow('.', '*') == '.');
        assert(f.narrow('a', '*') == 'a');
        assert(f.narrow('1', '*') == '1');
    }

  return 0;
}
