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

// explicit basic_ofstream(const wchar_t* s, ios_base::openmode mode = ios_base::out);

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
        std::ofstream fs(temp.c_str());
        fs << 3.25;
    }
    {
        std::ifstream fs(temp.c_str());
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    {
        std::ifstream fs(temp.c_str(), std::ios_base::out);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    _wremove(temp.c_str());
    {
        std::wofstream fs(temp.c_str());
        fs << 3.25;
    }
    {
        std::wifstream fs(temp.c_str());
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    {
        std::wifstream fs(temp.c_str(), std::ios_base::out);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    _wremove(temp.c_str());

    return 0;
}
