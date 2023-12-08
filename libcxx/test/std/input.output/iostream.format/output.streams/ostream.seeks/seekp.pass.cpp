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

// basic_ostream<charT,traits>& seekp(pos_type pos);

#include <ostream>
#include <cassert>

#include "test_macros.h"

int seekpos_called = 0;

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_streambuf<CharT> base;
    testbuf() {}

protected:

    typename base::pos_type
    seekpos(typename base::pos_type sp, std::ios_base::openmode which)
    {
        ++seekpos_called;
        assert(which == std::ios_base::out);
        return sp;
    }
};

int main(int, char**)
{
    {
        seekpos_called = 0;
        std::ostream os((std::streambuf*)0);
        assert(&os.seekp(5) == &os);
        assert(seekpos_called == 0);
    }
    {
        seekpos_called = 0;
        testbuf<char> sb;
        std::ostream os(&sb);
        assert(&os.seekp(10) == &os);
        assert(seekpos_called == 1);
        assert(os.good());
        assert(&os.seekp(-1) == &os);
        assert(seekpos_called == 2);
        assert(os.fail());
    }
    { // See https://llvm.org/PR21361
        seekpos_called = 0;
        testbuf<char> sb;
        std::ostream os(&sb);
        os.setstate(std::ios_base::eofbit);
        assert(&os.seekp(10) == &os);
        assert(seekpos_called == 1);
        assert(os.rdstate() == std::ios_base::eofbit);
    }

  return 0;
}
