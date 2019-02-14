//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <string>

// basic_string(initializer_list<charT> il, const Allocator& a = Allocator());

#include <string>
#include <cassert>

#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::string s = {'a', 'b', 'c'};
        assert(s == "abc");
    }
    {
        std::wstring s;
        s = {L'a', L'b', L'c'};
        assert(s == L"abc");
    }
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
        S s = {'a', 'b', 'c'};
        assert(s == "abc");
    }
    {
        typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, min_allocator<wchar_t>> S;
        S s;
        s = {L'a', L'b', L'c'};
        assert(s == L"abc");
    }

  return 0;
}
