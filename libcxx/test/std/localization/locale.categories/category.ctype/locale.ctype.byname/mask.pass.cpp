//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype_byname;

// bool is(mask m, charT c) const;

#include <locale>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l("C");
    {
        typedef std::ctype<char> CF;
        const CF& cf = std::use_facet<CF>(l);

        // NOTE:
        // The ctype masks in Newlib don't form a proper bitmask because
        // the mask is only 8 bits wide, and there are more than 8 mask
        // kinds. This means that the mask for alpha is (_U | _L), which
        // is tricky to match in the do_is implementation because in
        // [22.4.1.1.2 2] the standard specifies that the match code behaves
        // like (m & M) != 0, but following this exactly would give false
        // positives for characters that are both 'upper' and 'alpha', but
        // not 'lower', for example.
        assert( cf.is(CF::upper,  'A'));
        assert(!cf.is(CF::lower,  'A'));
        assert( cf.is(CF::alpha,  'A'));

        assert(!cf.is(CF::upper,  'a'));
        assert( cf.is(CF::lower,  'a'));
        assert( cf.is(CF::alpha,  'a'));
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        typedef std::ctype<wchar_t> WF;
        const WF& wf = std::use_facet<WF>(l);

        assert( wf.is(WF::upper, L'A'));
        assert(!wf.is(WF::lower, L'A'));
        assert( wf.is(WF::alpha, L'A'));

        assert(!wf.is(WF::upper, L'a'));
        assert( wf.is(WF::lower, L'a'));
        assert( wf.is(WF::alpha, L'a'));
    }
#endif

  return 0;
}
