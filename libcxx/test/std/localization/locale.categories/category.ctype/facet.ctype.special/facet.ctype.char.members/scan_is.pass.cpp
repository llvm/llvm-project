//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>;

// const char* scan_is(mask m, const char* low, const char* high) const;

#include <locale>
#include <string>
#include <vector>
#include <cassert>

#include <stdio.h>

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<char> F;
        const F& f = std::use_facet<F>(l);
        const std::string in(" A\x07.a1");
        std::vector<F::mask> m(in.size());
        assert(f.scan_is(F::space, in.data(), in.data() + in.size()) - in.data() == 0);
        assert(f.scan_is(F::print, in.data(), in.data() + in.size()) - in.data() == 0);
        assert(f.scan_is(F::cntrl, in.data(), in.data() + in.size()) - in.data() == 2);
        assert(f.scan_is(F::upper, in.data(), in.data() + in.size()) - in.data() == 1);
        assert(f.scan_is(F::lower, in.data(), in.data() + in.size()) - in.data() == 4);
        assert(f.scan_is(F::alpha, in.data(), in.data() + in.size()) - in.data() == 1);
        assert(f.scan_is(F::digit, in.data(), in.data() + in.size()) - in.data() == 5);
        assert(f.scan_is(F::punct, in.data(), in.data() + in.size()) - in.data() == 3);
        assert(f.scan_is(F::xdigit, in.data(), in.data() + in.size()) - in.data() == 1);
        assert(f.scan_is(F::blank, in.data(), in.data() + in.size()) - in.data() == 0);
        assert(f.scan_is(F::alnum, in.data(), in.data() + in.size()) - in.data() == 1);
        assert(f.scan_is(F::graph, in.data(), in.data() + in.size()) - in.data() == 1);
    }

  return 0;
}
