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

// explicit basic_ofstream(const char* s, ios_base::openmode mode = ios_base::out);

// In C++23 and later, this test requires support for P2467R1 in the dylib (a3f17ba3febbd546f2342ffc780ac93b694fdc8d)
// XFAIL: (!c++03 && !c++11 && !c++14 && !c++17 && !c++20) && using-built-library-before-llvm-18

// XFAIL: LIBCXX-AIX-FIXME

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
        std::ofstream fs(temp.c_str());
        fs << 3.25;
    }
    {
        std::ifstream fs(temp.c_str());
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::basic_ofstream<char, operator_hijacker_char_traits<char> > fs(temp.c_str());
      fs << "3.25";
    }
    {
      std::ifstream fs(temp.c_str());
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::ofstream fs(temp.c_str(), std::ios_base::out);
      fs << 3.25;
    }
    {
      std::ifstream fs(temp.c_str());
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::basic_ofstream<char, operator_hijacker_char_traits<char> > fs(temp.c_str(), std::ios_base::out);
      fs << "3.25";
    }
    {
      std::ifstream fs(temp.c_str());
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
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
    std::remove(temp.c_str());

    {
      std::basic_ofstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fs(temp.c_str());
      fs << L"3.25";
    }
    {
      std::ifstream fs(temp.c_str());
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::wofstream fs(temp.c_str(), std::ios_base::out);
      fs << 3.25;
    }
    {
      std::wifstream fs(temp.c_str());
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());

    {
      std::basic_ofstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fs(temp.c_str(), std::ios_base::out);
      fs << L"3.25";
    }
    {
      std::ifstream fs(temp.c_str());
      double x = 0;
      fs >> x;
      assert(x == 3.25);
    }
    std::remove(temp.c_str());

#endif

#if TEST_STD_VER >= 23
    // Test all the noreplace flag combinations
    {
        std::ios_base::openmode modes[] = {
            std::ios_base::out | std::ios_base::noreplace,
            std::ios_base::out | std::ios_base::trunc | std::ios_base::noreplace,
            std::ios_base::in | std::ios_base::out | std::ios_base::trunc | std::ios_base::noreplace,
            std::ios_base::out | std::ios_base::noreplace | std::ios_base::binary,
            std::ios_base::out | std::ios_base::trunc | std::ios_base::noreplace | std::ios_base::binary,
            std::ios_base::in | std::ios_base::out | std::ios_base::trunc | std::ios_base::noreplace |
                std::ios_base::binary,
        };
        for (auto mode : modes) {
          std::string tmp = get_temp_file_name(); // also creates the file

          {
            std::ofstream f(tmp.c_str(), mode);
            assert(!f.is_open()); // since it already exists
          }

          {
            std::remove(tmp.c_str());

            std::ofstream f(tmp.c_str(), mode);
            assert(f.is_open()); // since it doesn't exist
          }
        }

#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
        for (auto mode : modes) {
          std::string tmp = get_temp_file_name(); // also creates the file

          {
            std::wofstream f(tmp.c_str(), mode);
            assert(!f.is_open()); // since it already exists
          }

          {
            std::remove(tmp.c_str());

            std::wofstream f(tmp.c_str(), mode);
            assert(f.is_open()); // since it doesn't exist
          }
        }
#  endif
    }
#endif // TEST_STD_VER >= 23

    return 0;
}
