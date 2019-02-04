//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test:

// template <class charT, class traits, size_t N>
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is, bitset<N>& x);

#include <bitset>
#include <sstream>
#include <cassert>

int main()
{
    std::ostringstream os;
    std::bitset<8> b(0x5A);
    os << b;
    assert(os.str() == "01011010");
}
