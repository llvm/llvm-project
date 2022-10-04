//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_fstream

// void open(const wchar_t* s, ios_base::openmode mode = ios_base::in|ios_base::out);

// This extension is only provided on Windows.
// REQUIRES: windows
// UNSUPPORTED: no-wide-characters

#include <fstream>
#include <cassert>
#include "test_macros.h"
#include "wide_temp_file.h"

int main(int, char**) {
    std::wstring temp = get_wide_temp_file_name();
    {
        std::fstream fs;
        assert(!fs.is_open());
        fs.open(temp.c_str(), std::ios_base::in | std::ios_base::out
                                        | std::ios_base::trunc);
        assert(fs.is_open());
        double x = 0;
        fs << 3.25;
        fs.seekg(0);
        fs >> x;
        assert(x == 3.25);
    }
    _wremove(temp.c_str());
    {
        std::wfstream fs;
        assert(!fs.is_open());
        fs.open(temp.c_str(), std::ios_base::in | std::ios_base::out
                                        | std::ios_base::trunc);
        assert(fs.is_open());
        double x = 0;
        fs << 3.25;
        fs.seekg(0);
        fs >> x;
        assert(x == 3.25);
    }
    _wremove(temp.c_str());

    return 0;
}
