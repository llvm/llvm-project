//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ifstream

// void open(const wchar_t* s, ios_base::openmode mode = ios_base::in);

#include <fstream>
#include <cassert>

int main(int, char**)
{
#ifdef _LIBCPP_HAS_OPEN_WITH_WCHAR
    {
        std::ifstream fs;
        assert(!fs.is_open());
        char c = 'a';
        fs >> c;
        assert(fs.fail());
        assert(c == 'a');
        fs.open(L"test.dat");
        assert(fs.is_open());
        fs >> c;
        assert(c == 'r');
    }
    {
        std::wifstream fs;
        assert(!fs.is_open());
        wchar_t c = L'a';
        fs >> c;
        assert(fs.fail());
        assert(c == L'a');
        fs.open(L"test.dat");
        assert(fs.is_open());
        fs >> c;
        assert(c == L'r');
    }
#endif

  return 0;
}
