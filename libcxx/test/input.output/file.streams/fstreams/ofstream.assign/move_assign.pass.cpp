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

// basic_ofstream& operator=(basic_ofstream&& rhs);

#include <fstream>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    char temp[L_tmpnam];
    tmpnam(temp);
    {
        std::ofstream fso(temp);
        std::ofstream fs;
        fs = move(fso);
        fs << 3.25;
    }
    {
        std::ifstream fs(temp);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    remove(temp);
    {
        std::wofstream fso(temp);
        std::wofstream fs;
        fs = move(fso);
        fs << 3.25;
    }
    {
        std::wifstream fs(temp);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    remove(temp);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
