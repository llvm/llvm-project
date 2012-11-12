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
// class basic_fstream

// void close();

#include <fstream>
#include <cassert>

int main()
{
    char temp[L_tmpnam];
    tmpnam(temp);
    {
        std::fstream fs;
        assert(!fs.is_open());
        fs.open(temp, std::ios_base::out);
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
    remove(temp);
    {
        std::wfstream fs;
        assert(!fs.is_open());
        fs.open(temp, std::ios_base::out);
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
    remove(temp);
}
