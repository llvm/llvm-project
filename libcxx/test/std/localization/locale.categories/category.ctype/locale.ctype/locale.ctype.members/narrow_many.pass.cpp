//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype;

// const charT* narrow(const charT* low, const charT*, char dfault, char* to) const;

#include <locale>
#include <string>
#include <vector>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<wchar_t> F;
        const F& f = std::use_facet<F>(l);
        std::wstring in(L" A\x07.a1");
        std::vector<char> v(in.size());

        assert(f.narrow(&in[0], in.data() + in.size(), '*', v.data()) == in.data() + in.size());
        assert(v[0] == ' ');
        assert(v[1] == 'A');
        assert(v[2] == '\x07');
        assert(v[3] == '.');
        assert(v[4] == 'a');
        assert(v[5] == '1');
    }

  return 0;
}
