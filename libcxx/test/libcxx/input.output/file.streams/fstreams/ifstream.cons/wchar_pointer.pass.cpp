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

// explicit basic_ifstream(const wchar_t* s, ios_base::openmode mode = ios_base::in);

#include <fstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#ifdef _LIBCPP_HAS_OPEN_WITH_WCHAR
    {
        std::ifstream fs(L"test.dat");
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    // std::ifstream(const wchar_t*, std::ios_base::openmode) is tested in
    // test/libcxx/input.output/file.streams/fstreams/ofstream.cons/wchar_pointer.pass.cpp
    // which creates writable files.
    {
        std::wifstream fs(L"test.dat");
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    // std::wifstream(const wchar_t*, std::ios_base::openmode) is tested in
    // test/libcxx/input.output/file.streams/fstreams/ofstream.cons/wchar_pointer.pass.cpp
    // which creates writable files.
#endif

  return 0;
}
