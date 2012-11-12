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

// void open(const string& s, ios_base::openmode mode = ios_base::out);

#include <fstream>
#include <cassert>

int main()
{
    char temp[L_tmpnam];
    tmpnam(temp);
    {
        std::ofstream fs;
        assert(!fs.is_open());
        char c = 'a';
        fs << c;
        assert(fs.fail());
        fs.open(std::string(temp));
        assert(fs.is_open());
        fs << c;
    }
    {
        std::ifstream fs(temp);
        char c = 0;
        fs >> c;
        assert(c == 'a');
    }
    remove(temp);
    {
        std::wofstream fs;
        assert(!fs.is_open());
        wchar_t c = L'a';
        fs << c;
        assert(fs.fail());
        fs.open(std::string(temp));
        assert(fs.is_open());
        fs << c;
    }
    {
        std::wifstream fs(temp);
        wchar_t c = 0;
        fs >> c;
        assert(c == L'a');
    }
    remove(temp);
}
