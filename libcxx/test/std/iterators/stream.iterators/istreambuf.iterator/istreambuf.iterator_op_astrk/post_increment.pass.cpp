//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// istreambuf_iterator

// proxy istreambuf_iterator<charT,traits>::operator++(int);

#include <iterator>
#include <sstream>
#include <cassert>

int main()
{
    {
        std::istringstream inf("abc");
        std::istreambuf_iterator<char> i(inf);
        assert(*i++ == 'a');
        assert(*i++ == 'b');
        assert(*i++ == 'c');
        assert(i == std::istreambuf_iterator<char>());
    }
    {
        std::wistringstream inf(L"abc");
        std::istreambuf_iterator<wchar_t> i(inf);
        assert(*i++ == L'a');
        assert(*i++ == L'b');
        assert(*i++ == L'c');
        assert(i == std::istreambuf_iterator<wchar_t>());
    }
}
