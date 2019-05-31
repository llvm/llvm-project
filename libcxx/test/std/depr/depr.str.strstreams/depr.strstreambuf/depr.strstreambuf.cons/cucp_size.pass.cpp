//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstreambuf

// strstreambuf(const unsigned char* gnext_arg, streamsize n);

#include <strstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        unsigned char buf[] = "abcd";
        std::strstreambuf sb(buf, sizeof(buf));
        assert(sb.sgetc() == 'a');
        assert(sb.snextc() == 'b');
        assert(sb.snextc() == 'c');
        assert(sb.snextc() == 'd');
        assert(sb.snextc() == 0);
        assert(sb.snextc() == EOF);
    }
    {
        unsigned char buf[] = "abcd";
        std::strstreambuf sb(buf, 0);
        assert(sb.sgetc() == 'a');
        assert(sb.snextc() == 'b');
        assert(sb.snextc() == 'c');
        assert(sb.snextc() == 'd');
        assert(sb.snextc() == EOF);
    }

  return 0;
}
