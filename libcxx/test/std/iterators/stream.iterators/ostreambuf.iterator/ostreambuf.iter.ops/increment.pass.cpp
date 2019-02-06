//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostreambuf_iterator

// ostreambuf_iterator<charT,traits>& operator++();
// ostreambuf_iterator<charT,traits>& operator++(int);

#include <iterator>
#include <sstream>
#include <cassert>

int main()
{
    {
        std::ostringstream outf;
        std::ostreambuf_iterator<char> i(outf);
        std::ostreambuf_iterator<char>& iref = ++i;
        assert(&iref == &i);
        std::ostreambuf_iterator<char>& iref2 = i++;
        assert(&iref2 == &i);
    }
    {
        std::wostringstream outf;
        std::ostreambuf_iterator<wchar_t> i(outf);
        std::ostreambuf_iterator<wchar_t>& iref = ++i;
        assert(&iref == &i);
        std::ostreambuf_iterator<wchar_t>& iref2 = i++;
        assert(&iref2 == &i);
    }
}
