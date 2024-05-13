//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ostream>

// template <class charT, class traits = char_traits<charT> >
//   class basic_ostream;

// template <class charT, class traits>
//   basic_ostream<charT,traits>& flush(basic_ostream<charT,traits>& os);

#include <ostream>
#include <cassert>

#include "test_macros.h"

int sync_called = 0;

template <class CharT>
class testbuf
    : public std::basic_streambuf<CharT>
{
public:
    testbuf()
    {
    }

protected:

    virtual int
        sync()
        {
            ++sync_called;
            return 0;
        }
};

int main(int, char**)
{
    {
        testbuf<char> sb;
        std::ostream os(&sb);
        std::flush(os);
        assert(sync_called == 1);
        assert(os.good());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb;
        std::wostream os(&sb);
        std::flush(os);
        assert(sync_called == 2);
        assert(os.good());
    }
#endif

  return 0;
}
