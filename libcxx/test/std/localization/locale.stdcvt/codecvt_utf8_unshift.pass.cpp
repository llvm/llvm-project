//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <codecvt>

// template <class Elem, unsigned long Maxcode = 0x10ffff,
//           codecvt_mode Mode = (codecvt_mode)0>
// class codecvt_utf8
//     : public codecvt<Elem, char, mbstate_t>
// {
//     // unspecified
// };

// result
//     unshift(stateT& state,
//             externT* to, externT* to_end, externT*& to_next) const;

#include <codecvt>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::codecvt_utf8<wchar_t> C;
        C c;
        char n[4] = {0};
        std::mbstate_t m;
        char* np = nullptr;
        std::codecvt_base::result r = c.unshift(m, n, n+4, np);
        assert(r == std::codecvt_base::noconv);
    }
    {
        typedef std::codecvt_utf8<char16_t> C;
        C c;
        char n[4] = {0};
        std::mbstate_t m;
        char* np = nullptr;
        std::codecvt_base::result r = c.unshift(m, n, n+4, np);
        assert(r == std::codecvt_base::noconv);
    }
    {
        typedef std::codecvt_utf8<char32_t> C;
        C c;
        char n[4] = {0};
        std::mbstate_t m;
        char* np = nullptr;
        std::codecvt_base::result r = c.unshift(m, n, n+4, np);
        assert(r == std::codecvt_base::noconv);
    }

  return 0;
}
