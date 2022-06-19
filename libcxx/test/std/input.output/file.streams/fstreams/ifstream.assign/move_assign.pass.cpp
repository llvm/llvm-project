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

// basic_ifstream& operator=(basic_ifstream&& rhs);

#include <fstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::ifstream fso("test.dat");
        std::ifstream fs;
        fs = std::move(fso);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wifstream fso("test.dat");
        std::wifstream fs;
        fs = std::move(fso);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
#endif

  return 0;
}
