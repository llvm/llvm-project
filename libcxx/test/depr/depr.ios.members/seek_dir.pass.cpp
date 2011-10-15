//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>
//
// class ios_base
// {
// public:
//     typedef T3 seek_dir;
// };

#include <strstream>
#include <cassert>

int main()
{
    std::strstream::seek_dir b = std::strstream::cur;
    assert(b == std::ios::cur);
}
