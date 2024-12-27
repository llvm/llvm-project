//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FILE_DEPENDENCIES: test.dat

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ifstream

// explicit basic_ifstream(const string& s, ios_base::openmode mode = ios_base::in);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <fstream>
#include <cassert>

#include "test_macros.h"
#include "operator_hijacker.h"

int main(int, char**)
{
    {
        std::ifstream fs(std::string("test.dat"));
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    {
      std::basic_ifstream<char, operator_hijacker_char_traits<char> > fs(std::string("test.dat"));
      std::basic_string<char, operator_hijacker_char_traits<char> > x;
      fs >> x;
      assert(x == "3.25");
    }
    // std::ifstream(const std::string&, std::ios_base::openmode) is tested in
    // test/std/input.output/file.streams/fstreams/ofstream.cons/string.pass.cpp
    // which creates writable files.

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wifstream fs(std::string("test.dat"));
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    {
      std::basic_ifstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fs(std::string("test.dat"));
      std::basic_string<wchar_t, operator_hijacker_char_traits<wchar_t> > x;
      fs >> x;
      assert(x == L"3.25");
    }
    // std::wifstream(const std::string&, std::ios_base::openmode) is tested in
    // test/std/input.output/file.streams/fstreams/ofstream.cons/string.pass.cpp
    // which creates writable files.
#endif

  return 0;
}
