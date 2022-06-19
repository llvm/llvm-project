//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_filebuf

// basic_filebuf& operator=(basic_filebuf&& rhs);

#include <fstream>
#include <cassert>
#include "test_macros.h"
#include "platform_support.h"

int main(int, char**)
{
    std::string temp = get_temp_file_name();
    {
        std::filebuf f;
        assert(f.open(temp.c_str(), std::ios_base::out | std::ios_base::in
                                               | std::ios_base::trunc) != 0);
        assert(f.is_open());
        assert(f.sputn("123", 3) == 3);
        f.pubseekoff(1, std::ios_base::beg);
        assert(f.sgetc() == '2');
        std::filebuf f2;
        f2 = std::move(f);
        assert(!f.is_open());
        assert(f2.is_open());
        assert(f2.sgetc() == '2');
    }
    std::remove(temp.c_str());

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wfilebuf f;
        assert(f.open(temp.c_str(), std::ios_base::out | std::ios_base::in
                                               | std::ios_base::trunc) != 0);
        assert(f.is_open());
        assert(f.sputn(L"123", 3) == 3);
        f.pubseekoff(1, std::ios_base::beg);
        assert(f.sgetc() == L'2');
        std::wfilebuf f2;
        f2 = std::move(f);
        assert(!f.is_open());
        assert(f2.is_open());
        assert(f2.sgetc() == L'2');
    }
    std::remove(temp.c_str());
#endif

  return 0;
}
