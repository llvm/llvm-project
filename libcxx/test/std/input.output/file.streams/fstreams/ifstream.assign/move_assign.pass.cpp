//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
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
        fs = move(fso);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    {
        std::wifstream fso("test.dat");
        std::wifstream fs;
        fs = move(fso);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }

  return 0;
}
