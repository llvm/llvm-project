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

// basic_fstream(basic_fstream&& rhs);

#include <fstream>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h"
#include "operator_hijacker.h"

int main(int, char**)
{
    std::string temp = get_temp_file_name();
    {
        std::fstream fso(temp, std::ios_base::in | std::ios_base::out
                                                 | std::ios_base::trunc);
        std::fstream fs = std::move(fso);
        double x = 0;
        fs << 3.25;
        fs.seekg(0);
        fs >> x;
        assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::basic_fstream<char, operator_hijacker_char_traits<char> > fso(
          temp, std::ios_base::in | std::ios_base::out | std::ios_base::trunc);
      std::basic_fstream<char, operator_hijacker_char_traits<char> > fs = std::move(fso);
      std::basic_string<char, operator_hijacker_char_traits<char> > x;
      fs << "3.25";
      fs.seekg(0);
      fs >> x;
      assert(x == "3.25");
    }
    std::remove(temp.c_str());

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wfstream fso(temp, std::ios_base::in | std::ios_base::out
                                                  | std::ios_base::trunc);
        std::wfstream fs = std::move(fso);
        double x = 0;
        fs << 3.25;
        fs.seekg(0);
        fs >> x;
        assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::basic_fstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fso(
          temp, std::ios_base::in | std::ios_base::out | std::ios_base::trunc);
      std::basic_fstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fs = std::move(fso);
      std::basic_string<wchar_t, operator_hijacker_char_traits<wchar_t> > x;
      fs << L"3.25";
      fs.seekg(0);
      fs >> x;
      assert(x == L"3.25");
    }
    std::remove(temp.c_str());
#endif

  return 0;
}
