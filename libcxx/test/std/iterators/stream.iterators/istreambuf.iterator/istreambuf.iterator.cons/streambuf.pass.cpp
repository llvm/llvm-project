//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// istreambuf_iterator

// istreambuf_iterator(basic_streambuf<charT,traits>* s) throw();

#include <iterator>
#include <sstream>
#include <cassert>

int main(int, char**)
{
    {
        std::istreambuf_iterator<char> i(nullptr);
        assert(i == std::istreambuf_iterator<char>());
    }
    {
        std::istringstream inf;
        std::istreambuf_iterator<char> i(inf.rdbuf());
        assert(i == std::istreambuf_iterator<char>());
    }
    {
        std::istringstream inf("a");
        std::istreambuf_iterator<char> i(inf.rdbuf());
        assert(i != std::istreambuf_iterator<char>());
    }
    {
        std::istreambuf_iterator<wchar_t> i(nullptr);
        assert(i == std::istreambuf_iterator<wchar_t>());
    }
    {
        std::wistringstream inf;
        std::istreambuf_iterator<wchar_t> i(inf.rdbuf());
        assert(i == std::istreambuf_iterator<wchar_t>());
    }
    {
        std::wistringstream inf(L"a");
        std::istreambuf_iterator<wchar_t> i(inf.rdbuf());
        assert(i != std::istreambuf_iterator<wchar_t>());
    }

  return 0;
}
