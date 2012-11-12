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

// explicit basic_ofstream(const string& s, ios_base::openmode mode = ios_base::out);

#include <fstream>
#include <cassert>

int main()
{
    char temp[L_tmpnam];
    tmpnam(temp);
    {
        std::ofstream fs((std::string(temp)));
        fs << 3.25;
    }
    {
        std::ifstream fs((std::string(temp)));
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    remove(temp);
    {
        std::wofstream fs((std::string(temp)));
        fs << 3.25;
    }
    {
        std::wifstream fs((std::string(temp)));
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    remove(temp);
}
