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

int main()
{
    char temp1[L_tmpnam], temp2[L_tmpnam];
    tmpnam(temp1);
    tmpnam(temp2);
    {
        std::ofstream fs1(temp1);
        std::ofstream fs2(temp2);
        fs1 << 3.25;
        fs2 << 4.5;
        fs1.swap(fs2);
        fs1 << ' ' << 3.25;
        fs2 << ' ' << 4.5;
    }
    {
        std::ifstream fs(temp1);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
        fs >> x;
        assert(x == 4.5);
    }
    remove(temp1);
    {
        std::ifstream fs(temp2);
        double x = 0;
        fs >> x;
        assert(x == 4.5);
        fs >> x;
        assert(x == 3.25);
    }
    remove(temp2);
    {
        std::wofstream fs1(temp1);
        std::wofstream fs2(temp2);
        fs1 << 3.25;
        fs2 << 4.5;
        fs1.swap(fs2);
        fs1 << ' ' << 3.25;
        fs2 << ' ' << 4.5;
    }
    {
        std::wifstream fs(temp1);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
        fs >> x;
        assert(x == 4.5);
    }
    remove(temp1);
    {
        std::wifstream fs(temp2);
        double x = 0;
        fs >> x;
        assert(x == 4.5);
        fs >> x;
        assert(x == 3.25);
    }
    remove(temp2);
}
