//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <codecvt>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX26_REMOVED_CODECVT

// template <class Elem, unsigned long Maxcode = 0x10ffff,
//           codecvt_mode Mode = (codecvt_mode)0>
// class codecvt_utf16
//     : public codecvt<Elem, char, mbstate_t>
// {
//     // unspecified
// };

// bool always_noconv() const throw();

#include <codecvt>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        typedef std::codecvt_utf16<wchar_t> C;
        C c;
        bool r = c.always_noconv();
        assert(r == false);
    }
#endif
    {
        typedef std::codecvt_utf16<char16_t> C;
        C c;
        bool r = c.always_noconv();
        assert(r == false);
    }
    {
        typedef std::codecvt_utf16<char32_t> C;
        C c;
        bool r = c.always_noconv();
        assert(r == false);
    }

  return 0;
}
