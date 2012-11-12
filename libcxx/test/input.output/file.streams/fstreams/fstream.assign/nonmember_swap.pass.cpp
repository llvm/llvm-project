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

// template <class charT, class traits>
//   void swap(basic_fstream<charT, traits>& x, basic_fstream<charT, traits>& y);

#include <fstream>
#include <cassert>

int main()
{
    char temp1[L_tmpnam], temp2[L_tmpnam];
    tmpnam(temp1);
    tmpnam(temp2);
    {
        std::fstream fs1(temp1, std::ios_base::in | std::ios_base::out
                                                  | std::ios_base::trunc);
        std::fstream fs2(temp2, std::ios_base::in | std::ios_base::out
                                                  | std::ios_base::trunc);
        fs1 << 1 << ' ' << 2;
        fs2 << 2 << ' ' << 1;
        fs1.seekg(0);
        swap(fs1, fs2);
        fs1.seekg(0);
        int i;
        fs1 >> i;
        assert(i == 2);
        fs1 >> i;
        assert(i == 1);
        i = 0;
        fs2 >> i;
        assert(i == 1);
        fs2 >> i;
        assert(i == 2);
    }
    std::remove(temp1);
    std::remove(temp2);
    {
        std::wfstream fs1(temp1, std::ios_base::in | std::ios_base::out
                                                   | std::ios_base::trunc);
        std::wfstream fs2(temp2, std::ios_base::in | std::ios_base::out
                                                   | std::ios_base::trunc);
        fs1 << 1 << ' ' << 2;
        fs2 << 2 << ' ' << 1;
        fs1.seekg(0);
        swap(fs1, fs2);
        fs1.seekg(0);
        int i;
        fs1 >> i;
        assert(i == 2);
        fs1 >> i;
        assert(i == 1);
        i = 0;
        fs2 >> i;
        assert(i == 1);
        fs2 >> i;
        assert(i == 2);
    }
    std::remove(temp1);
    std::remove(temp2);
}
