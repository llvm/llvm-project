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

// explicit basic_ofstream(const string& s, ios_base::openmode mode = ios_base::out);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <fstream>
#include <cassert>
#include <ios>

#include "test_macros.h"
#include "operator_hijacker.h"
#include "platform_support.h"

int main(int, char**)
{
    std::string temp = get_temp_file_name();

    {
        std::ofstream fs(temp);
        fs << 3.25;
    }
    {
        std::ifstream fs(temp);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::basic_ofstream<char, operator_hijacker_char_traits<char> > fs(temp);
      fs << "3.25";
    }
    {
      std::ifstream fs(temp);
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::ofstream fs(temp, std::ios_base::out);
      fs << 3.25;
    }
    {
      std::ifstream fs(temp);
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::basic_ofstream<char, operator_hijacker_char_traits<char> > fs(temp, std::ios_base::out);
      fs << "3.25";
    }
    {
      std::ifstream fs(temp);
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wofstream fs(temp);
        fs << 3.25;
    }
    {
        std::wifstream fs(temp);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::basic_ofstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fs(temp);
      fs << L"3.25";
    }
    {
      std::wifstream fs(temp);
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::wofstream fs(temp, std::ios_base::out);
      fs << 3.25;
    }
    {
      std::wifstream fs(temp);
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::basic_ofstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fs(temp, std::ios_base::out);
      fs << L"3.25";
    }
    {
      std::wifstream fs(temp);
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());
#endif

  return 0;
}
