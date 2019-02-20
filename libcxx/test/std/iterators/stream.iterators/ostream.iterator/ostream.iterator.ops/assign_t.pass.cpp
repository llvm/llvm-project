//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostream_iterator

// ostream_iterator& operator=(const T& value);

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

#if defined(TEST_COMPILER_CLANG)
#pragma clang diagnostic ignored "-Wliteral-conversion"
#endif

#ifdef TEST_COMPILER_C1XX
#pragma warning(disable: 4244) // conversion from 'X' to 'Y', possible loss of data
#endif

int main(int, char**)
{
    {
        std::ostringstream outf;
        std::ostream_iterator<int> i(outf);
        i = 2.4;
        assert(outf.str() == "2");
    }
    {
        std::ostringstream outf;
        std::ostream_iterator<int> i(outf, ", ");
        i = 2.4;
        assert(outf.str() == "2, ");
    }
    {
        std::wostringstream outf;
        std::ostream_iterator<int, wchar_t> i(outf);
        i = 2.4;
        assert(outf.str() == L"2");
    }
    {
        std::wostringstream outf;
        std::ostream_iterator<int, wchar_t> i(outf, L", ");
        i = 2.4;
        assert(outf.str() == L"2, ");
    }

  return 0;
}
