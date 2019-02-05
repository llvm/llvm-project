//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// basic_ofstream(basic_ofstream&& rhs);

#include <fstream>
#include <cassert>
#include "platform_support.h"

int main(int, char**)
{
    std::string temp = get_temp_file_name();
    {
        std::ofstream fso(temp.c_str());
        std::ofstream fs = move(fso);
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
        std::wofstream fso(temp.c_str());
        std::wofstream fs = move(fso);
        fs << 3.25;
    }
    {
        std::wifstream fs(temp.c_str());
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    std::remove(temp.c_str());

  return 0;
}
