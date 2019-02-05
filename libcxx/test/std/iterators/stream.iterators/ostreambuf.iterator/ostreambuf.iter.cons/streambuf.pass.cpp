//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostreambuf_iterator

// ostreambuf_iterator(streambuf_type* s) throw();

#include <iterator>
#include <sstream>
#include <cassert>

int main()
{
    {
        std::ostringstream outf;
        std::ostreambuf_iterator<char> i(outf.rdbuf());
        assert(!i.failed());
    }
    {
        std::wostringstream outf;
        std::ostreambuf_iterator<wchar_t> i(outf.rdbuf());
        assert(!i.failed());
    }
}
