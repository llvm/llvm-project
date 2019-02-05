//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype;

// const charT* do_is(const charT* low, const charT* high, mask* vec) const;

#include <locale>
#include <string>
#include <vector>
#include <cassert>

#include <stdio.h>

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<wchar_t> F;
        const F& f = std::use_facet<F>(l);
        const std::wstring in(L" A\x07.a1");
        std::vector<F::mask> m(in.size());
        const wchar_t* h = f.is(in.data(), in.data() + in.size(), m.data());
        assert(h == in.data() + in.size());

        // L' '
        assert( (m[0] & F::space));
        assert( (m[0] & F::print));
        assert(!(m[0] & F::cntrl));
        assert(!(m[0] & F::upper));
        assert(!(m[0] & F::lower));
        assert(!(m[0] & F::alpha));
        assert(!(m[0] & F::digit));
        assert(!(m[0] & F::punct));
        assert(!(m[0] & F::xdigit));
        assert( (m[0] & F::blank));
        assert(!(m[0] & F::alnum));
        assert(!(m[0] & F::graph));

        // L'A'
        assert(!(m[1] & F::space));
        assert( (m[1] & F::print));
        assert(!(m[1] & F::cntrl));
        assert( (m[1] & F::upper));
        assert(!(m[1] & F::lower));
        assert( (m[1] & F::alpha));
        assert(!(m[1] & F::digit));
        assert(!(m[1] & F::punct));
        assert( (m[1] & F::xdigit));
        assert(!(m[1] & F::blank));
        assert( (m[1] & F::alnum));
        assert( (m[1] & F::graph));

        // L'\x07'
        assert(!(m[2] & F::space));
        assert(!(m[2] & F::print));
        assert( (m[2] & F::cntrl));
        assert(!(m[2] & F::upper));
        assert(!(m[2] & F::lower));
        assert(!(m[2] & F::alpha));
        assert(!(m[2] & F::digit));
        assert(!(m[2] & F::punct));
        assert(!(m[2] & F::xdigit));
        assert(!(m[2] & F::blank));
        assert(!(m[2] & F::alnum));
        assert(!(m[2] & F::graph));

        // L'.'
        assert(!(m[3] & F::space));
        assert( (m[3] & F::print));
        assert(!(m[3] & F::cntrl));
        assert(!(m[3] & F::upper));
        assert(!(m[3] & F::lower));
        assert(!(m[3] & F::alpha));
        assert(!(m[3] & F::digit));
        assert( (m[3] & F::punct));
        assert(!(m[3] & F::xdigit));
        assert(!(m[3] & F::blank));
        assert(!(m[3] & F::alnum));
        assert( (m[3] & F::graph));

        // L'a'
        assert(!(m[4] & F::space));
        assert( (m[4] & F::print));
        assert(!(m[4] & F::cntrl));
        assert(!(m[4] & F::upper));
        assert( (m[4] & F::lower));
        assert( (m[4] & F::alpha));
        assert(!(m[4] & F::digit));
        assert(!(m[4] & F::punct));
        assert( (m[4] & F::xdigit));
        assert(!(m[4] & F::blank));
        assert( (m[4] & F::alnum));
        assert( (m[4] & F::graph));

        // L'1'
        assert(!(m[5] & F::space));
        assert( (m[5] & F::print));
        assert(!(m[5] & F::cntrl));
        assert(!(m[5] & F::upper));
        assert(!(m[5] & F::lower));
        assert(!(m[5] & F::alpha));
        assert( (m[5] & F::digit));
        assert(!(m[5] & F::punct));
        assert( (m[5] & F::xdigit));
        assert(!(m[5] & F::blank));
        assert( (m[5] & F::alnum));
        assert( (m[5] & F::graph));
    }

  return 0;
}
