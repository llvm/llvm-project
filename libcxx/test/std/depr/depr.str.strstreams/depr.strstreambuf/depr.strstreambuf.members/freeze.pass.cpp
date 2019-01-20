//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstreambuf

// void freeze(bool freezefl = true);

#include <strstream>
#include <cassert>

int main()
{
    {
        std::strstreambuf sb;
        sb.freeze(true);
        assert(sb.sputc('a') == EOF);
        sb.freeze(false);
        assert(sb.sputc('a') == 'a');
    }
}
