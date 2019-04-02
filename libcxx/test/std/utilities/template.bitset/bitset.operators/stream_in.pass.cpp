//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test:

// template <class charT, class traits, size_t N>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& os, const bitset<N>& x);

#include <bitset>
#include <sstream>
#include <cassert>

int main(int, char**)
{
    std::istringstream in("01011010");
    std::bitset<8> b;
    in >> b;
    assert(b.to_ulong() == 0x5A);

  return 0;
}
