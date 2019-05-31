//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostreambuf_iterator

// ostreambuf_iterator<charT,traits>&
//   operator=(charT c);

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::ostringstream outf;
        std::ostreambuf_iterator<char> i(outf);
        i = 'a';
        assert(outf.str() == "a");
        i = 'b';
        assert(outf.str() == "ab");
    }
    {
        std::wostringstream outf;
        std::ostreambuf_iterator<wchar_t> i(outf);
        i = L'a';
        assert(outf.str() == L"a");
        i = L'b';
        assert(outf.str() == L"ab");
    }

  return 0;
}
