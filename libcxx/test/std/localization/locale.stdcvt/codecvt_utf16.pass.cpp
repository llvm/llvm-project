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
// class codecvt_utf16
//     : public codecvt<Elem, char, mbstate_t>
// {
//     // unspecified
// };

// XFAIL: no-wide-characters

// Not a portable test

#include <codecvt>
#include <cstdlib>
#include <cassert>

#include "count_new.h"

#include "test_macros.h"

int main(int, char**)
{
    globalMemCounter.reset();
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
        typedef std::codecvt_utf16<wchar_t> C;
        C c;
        assert(globalMemCounter.checkOutstandingNewEq(0));
    }
    {
        typedef std::codecvt_utf16<wchar_t> C;
        std::locale loc(std::locale::classic(), new C);
        assert(globalMemCounter.checkOutstandingNewNotEq(0));
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
