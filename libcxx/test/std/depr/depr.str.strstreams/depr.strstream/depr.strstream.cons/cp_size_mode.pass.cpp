//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstream

// strstream(char* s, int n, ios_base::openmode mode = ios_base::in | ios_base::out);

#include <strstream>
#include <cassert>
#include <string>

#include "test_macros.h"

int main(int, char**)
{
    {
        char buf[] = "123 4.5 dog";
        std::strstream inout(buf, 0);
        assert(inout.str() == std::string("123 4.5 dog"));
        int i = 321;
        double d = 5.5;
        std::string s("cat");
        inout >> i;
        assert(inout.fail());
        inout.clear();
        inout << i << ' ' << d << ' ' << s;
        assert(inout.str() == std::string("321 5.5 cat"));
        i = 0;
        d = 0;
        s = "";
        inout >> i >> d >> s;
        assert(i == 321);
        assert(d == 5.5);
        assert(s == "cat");
    }
    {
        char buf[23] = "123 4.5 dog";
        std::strstream inout(buf, 11, std::ios::app);
        assert(inout.str() == std::string("123 4.5 dog"));
        int i = 0;
        double d = 0;
        std::string s;
        inout >> i >> d >> s;
        assert(i == 123);
        assert(d == 4.5);
        assert(s == "dog");
        i = 321;
        d = 5.5;
        s = "cat";
        inout.clear();
        inout << i << ' ' << d << ' ' << s;
        assert(inout.str() == std::string("123 4.5 dog321 5.5 cat"));
    }

  return 0;
}
