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

// XFAIL: LIBCXX-AIX-FIXME

#include <fstream>
#include <cassert>
#include "test_macros.h"
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
    {
        std::ifstream fs(temp.c_str(), std::ios_base::out);
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
    {
        std::wifstream fs(temp.c_str(), std::ios_base::out);
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
