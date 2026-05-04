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

// This extension is only provided on Windows.
// REQUIRES: windows
// UNSUPPORTED: no-wide-characters

// TODO: This should not be necessary
// ADDITIONAL_COMPILE_FLAGS:-D_LIBCPP_ENABLE_CXX26_REMOVED_CODECVT -D_LIBCPP_ENABLE_CXX26_REMOVED_WSTRING_CONVERT

#include <fstream>
#include <cassert>
#include "test_macros.h"
#include "wide_temp_file.h"

int main(int, char**) {
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

    return 0;
}
