//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<char16_t, char, mbstate_t>

// result out(stateT& state,
//            const internT* from, const internT* from_end, const internT*& from_next,
//            externT* to, externT* to_end, externT*& to_next) const;

#include <locale>
#include <string>
#include <vector>
#include <cassert>

#include <stdio.h>

#include "test_macros.h"

typedef std::codecvt<char16_t, char, std::mbstate_t> F;

int main(int, char**)
{
    std::locale l = std::locale::classic();
    const F& f = std::use_facet<F>(l);
    {
        F::intern_type from[9] = {'s', 'o', 'm', 'e', ' ', 't', 'e', 'x', 't'};
        char to[9] = {0};
        std::mbstate_t mbs = {};
        const F::intern_type* from_next = 0;
        char* to_next = 0;
        F::result r = f.out(mbs, from, from + 9, from_next,
                                 to, to + 9, to_next);
        assert(r == F::ok);
        assert(from_next - from == 9);
        assert(to_next - to == 9);
        for (unsigned i = 0; i < 9; ++i)
            assert(to[i] == from[i]);
    }

  return 0;
}
