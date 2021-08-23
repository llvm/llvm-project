//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype;

// const char* widen(const char* low, const char* high, charT* to) const;

// XFAIL: libcpp-has-no-wide-characters

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
        std::string in(" A\x07.a1");
        std::vector<wchar_t> v(in.size());

        assert(f.widen(&in[0], in.data() + in.size(), v.data()) == in.data() + in.size());
        assert(v[0] == L' ');
        assert(v[1] == L'A');
        assert(v[2] == L'\x07');
        assert(v[3] == L'.');
        assert(v[4] == L'a');
        assert(v[5] == L'1');
    }

  return 0;
}
