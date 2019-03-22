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

// int encoding() const throw();

#include <codecvt>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::codecvt_utf8<wchar_t> C;
        C c;
        int r = c.encoding();
        assert(r == 0);
    }
    {
        typedef std::codecvt_utf8<char16_t> C;
        C c;
        int r = c.encoding();
        assert(r == 0);
    }
    {
        typedef std::codecvt_utf8<char32_t> C;
        C c;
        int r = c.encoding();
        assert(r == 0);
    }

  return 0;
}
