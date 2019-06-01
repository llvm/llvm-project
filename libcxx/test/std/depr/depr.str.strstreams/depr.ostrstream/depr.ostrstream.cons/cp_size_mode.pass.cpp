//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class ostrstream

// ostrstream(char* s, int n, ios_base::openmode mode = ios_base::out);

#include <strstream>
#include <cassert>
#include <string>

#include "test_macros.h"

int main(int, char**)
{
    {
        char buf[] = "123 4.5 dog";
        std::ostrstream out(buf, 0);
        assert(out.str() == std::string("123 4.5 dog"));
        int i = 321;
        double d = 5.5;
        std::string s("cat");
        out << i << ' ' << d << ' ' << s << std::ends;
        assert(out.str() == std::string("321 5.5 cat"));
    }
    {
        char buf[23] = "123 4.5 dog";
        std::ostrstream out(buf, 11, std::ios::app);
        assert(out.str() == std::string("123 4.5 dog"));
        int i = 321;
        double d = 5.5;
        std::string s("cat");
        out << i << ' ' << d << ' ' << s << std::ends;
        assert(out.str() == std::string("123 4.5 dog321 5.5 cat"));
    }

  return 0;
}
