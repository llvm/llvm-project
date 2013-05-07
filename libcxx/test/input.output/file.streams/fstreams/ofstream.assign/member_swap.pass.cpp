//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// void swap(basic_ofstream& rhs);

#include <fstream>
#include <cassert>
#include "platform_support.h"

int main()
{
    std::string temp1 = get_temp_file_name();
    std::string temp2 = get_temp_file_name();
    {
        std::ofstream fs1(temp1.c_str());
        std::ofstream fs2(temp2.c_str());
        fs1 << 3.25;
        fs2 << 4.5;
        fs1.swap(fs2);
        fs1 << ' ' << 3.25;
        fs2 << ' ' << 4.5;
    }
    {
        std::ifstream fs(temp1.c_str());
        double x = 0;
        fs >> x;
        assert(x == 3.25);
        fs >> x;
        assert(x == 4.5);
    }
    std::remove(temp1.c_str());
    {
        std::ifstream fs(temp2.c_str());
        double x = 0;
        fs >> x;
        assert(x == 4.5);
        fs >> x;
        assert(x == 3.25);
    }
    std::remove(temp2.c_str());
    {
        std::wofstream fs1(temp1.c_str());
        std::wofstream fs2(temp2.c_str());
        fs1 << 3.25;
        fs2 << 4.5;
        fs1.swap(fs2);
        fs1 << ' ' << 3.25;
        fs2 << ' ' << 4.5;
    }
    {
        std::wifstream fs(temp1.c_str());
        double x = 0;
        fs >> x;
        assert(x == 3.25);
        fs >> x;
        assert(x == 4.5);
    }
    std::remove(temp1.c_str());
    {
        std::wifstream fs(temp2.c_str());
        double x = 0;
        fs >> x;
        assert(x == 4.5);
        fs >> x;
        assert(x == 3.25);
    }
    std::remove(temp2.c_str());
}
