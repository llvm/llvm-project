//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class ostrstream

// void freeze(bool freezefl = true);

#include <strstream>
#include <cassert>

int main()
{
    {
        std::ostrstream out;
        out.freeze();
        assert(!out.fail());
        out << 'a';
        assert(out.fail());
        out.clear();
        out.freeze(false);
        out << 'a';
        out << char(0);
        assert(out.str() == std::string("a"));
        out.freeze(false);
    }
}
