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

// void close();

#include <fstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::ifstream fs;
        assert(!fs.is_open());
        fs.open("test.dat");
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wifstream fs;
        assert(!fs.is_open());
        fs.open("test.dat");
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
#endif

  return 0;
}
