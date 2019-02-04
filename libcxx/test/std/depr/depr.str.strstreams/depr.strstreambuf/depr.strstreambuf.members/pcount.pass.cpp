//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstreambuf

// int pcount() const;

#include <strstream>
#include <cassert>

int main(int, char**)
{
    {
        std::strstreambuf sb;
        assert(sb.pcount() == 0);
        assert(sb.sputc('a') == 'a');
        assert(sb.pcount() == 1);
        assert(sb.sputc(0) == 0);
        assert(sb.pcount() == 2);
        assert(sb.str() == std::string("a"));
        assert(sb.pcount() == 2);
        sb.freeze(false);
    }

  return 0;
}
