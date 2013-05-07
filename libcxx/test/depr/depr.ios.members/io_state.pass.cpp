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
//     typedef T1 io_state;
// };

#include <strstream>
#include <cassert>

int main()
{
    std::strstream::io_state b = std::strstream::eofbit;
    assert(b == std::ios::eofbit);
}
