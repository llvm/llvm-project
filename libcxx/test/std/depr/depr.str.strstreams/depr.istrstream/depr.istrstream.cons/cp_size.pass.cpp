//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class istrstream

// explicit istrstream(char* s, streamsize n);

#include <strstream>
#include <cassert>
#include <string>

#include "test_macros.h"

int main(int, char**)
{
    {
        char buf[] = "123 4.5 dog";
        std::istrstream in(buf, 7);
        int i;
        in >> i;
        assert(i == 123);
        double d;
        in >> d;
        assert(d == 4.5);
        std::string s;
        in >> s;
        assert(s == "");
        assert(in.eof());
        assert(in.fail());
        in.clear();
        in.putback('5');
        assert(!in.fail());
        in.putback('5');
        assert(!in.fail());
        assert(buf[5] == '5');
        assert(buf[6] == '5');
    }

  return 0;
}
