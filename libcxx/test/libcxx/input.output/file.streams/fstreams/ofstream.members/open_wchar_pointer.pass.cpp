//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// void open(const wchar_t* s, ios_base::openmode mode = ios_base::out);

#include <fstream>
#include <cassert>
#include "platform_support.h"

int main()
{
#ifdef _LIBCPP_HAS_OPEN_WITH_WCHAR
    std::wstring temp = get_wide_temp_file_name();
    {
        std::ofstream fs;
        assert(!fs.is_open());
        char c = 'a';
        fs << c;
        assert(fs.fail());
        fs.open(temp.c_str());
        assert(fs.is_open());
        fs << c;
    }
    {
        std::ifstream fs(temp.c_str());
        char c = 0;
        fs >> c;
        assert(c == 'a');
    }
    _wremove(temp.c_str());
    {
        std::wofstream fs;
        assert(!fs.is_open());
        wchar_t c = L'a';
        fs << c;
        assert(fs.fail());
        fs.open(temp.c_str());
        assert(fs.is_open());
        fs << c;
    }
    {
        std::wifstream fs(temp.c_str());
        wchar_t c = 0;
        fs >> c;
        assert(c == L'a');
    }
    _wremove(temp.c_str());
#endif
}
