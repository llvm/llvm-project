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

// basic_ifstream(basic_ifstream&& rhs);

#include <fstream>
#include <cassert>
#include <ios>

#include "test_macros.h"
#include "operator_hijacker.h"

int main(int, char**)
{
    {
        std::ifstream fso("test.dat");
        std::ifstream fs = std::move(fso);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    {
      std::basic_ifstream<char, operator_hijacker_char_traits<char> > fso("test.dat");
      std::basic_ifstream<char, operator_hijacker_char_traits<char> > fs = std::move(fso);
      std::basic_string<char, operator_hijacker_char_traits<char> > x;
      fs >> x;
      assert(x == "3.25");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wifstream fso("test.dat");
        std::wifstream fs = std::move(fso);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    {
      std::basic_ifstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fso("test.dat");
      std::basic_ifstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fs = std::move(fso);
      std::basic_string<wchar_t, operator_hijacker_char_traits<wchar_t> > x;
      fs >> x;
      assert(x == L"3.25");
    }
#endif

  return 0;
}
