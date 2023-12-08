//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// basic_filebuf<charT,traits>* open(const char* s, ios_base::openmode mode);

// XFAIL: LIBCXX-AIX-FIXME

#include <fstream>
#include <cassert>
#include "test_macros.h"
#include "platform_support.h"

int main(int, char**)
{
    std::string temp = get_temp_file_name();
    {
        std::filebuf f;
        assert(f.open(temp.c_str(), std::ios_base::out) != 0);
        assert(f.is_open());
        assert(f.sputn("123", 3) == 3);
    }
    {
        std::filebuf f;
        assert(f.open(temp.c_str(), std::ios_base::in) != 0);
        assert(f.is_open());
        assert(f.sbumpc() == '1');
        assert(f.sbumpc() == '2');
        assert(f.sbumpc() == '3');
    }
    std::remove(temp.c_str());

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wfilebuf f;
        assert(f.open(temp.c_str(), std::ios_base::out) != 0);
        assert(f.is_open());
        assert(f.sputn(L"123", 3) == 3);
    }
    {
        std::wfilebuf f;
        assert(f.open(temp.c_str(), std::ios_base::in) != 0);
        assert(f.is_open());
        assert(f.sbumpc() == L'1');
        assert(f.sbumpc() == L'2');
        assert(f.sbumpc() == L'3');
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
            std::filebuf f;
            f.open(tmp.c_str(), mode);
            assert(!f.is_open()); // since it already exists
          }

          {
            std::remove(tmp.c_str());

            std::filebuf f;
            f.open(tmp.c_str(), mode);
            assert(f.is_open()); // since it doesn't exist
          }
        }

#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
        for (auto mode : modes) {
          std::string tmp = get_temp_file_name(); // also creates the file

          {
            std::wfilebuf f;
            f.open(tmp.c_str(), mode);
            assert(!f.is_open()); // since it already exists
          }

          {
            std::remove(tmp.c_str());

            std::wfilebuf f;
            f.open(tmp.c_str(), mode);
            assert(f.is_open()); // since it doesn't exist
          }
        }
#  endif
    }
#endif // TEST_STD_VER >= 23

    return 0;
}
