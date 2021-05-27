//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class charT, class traits = char_traits<charT> >
// class istreambuf_iterator {
// public:
//     ...
//     proxy operator++(int);

// class proxy
// {
// public:
//     charT operator*();
// };

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::istringstream inf("abc");
        std::istreambuf_iterator<char> i(inf);
        assert(*i++ == 'a');
    }
    {
        std::wistringstream inf(L"abc");
        std::istreambuf_iterator<wchar_t> i(inf);
        assert(*i++ == L'a');
    }

  return 0;
}
