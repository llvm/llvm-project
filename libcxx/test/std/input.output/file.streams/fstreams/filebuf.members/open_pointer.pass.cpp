//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// basic_filebuf<charT,traits>* open(const char* s, ios_base::openmode mode);

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
    remove(temp.c_str());

  return 0;
}
