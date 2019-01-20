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

// basic_ostream& flush();

#include <ostream>
#include <cassert>

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
            if (sync_called++ == 1)
                return -1;
            return 0;
        }
};

int main()
{
    {
        testbuf<char> sb;
        std::ostream os(&sb);
        os.flush();
        assert(os.good());
        assert(sync_called == 1);
        os.flush();
        assert(os.bad());
        assert(sync_called == 2);
    }
}
