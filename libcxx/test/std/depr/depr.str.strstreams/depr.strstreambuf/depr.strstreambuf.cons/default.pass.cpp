//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstreambuf

// explicit strstreambuf(streamsize alsize_arg = 0);

#include <strstream>
#include <cassert>

int main()
{
    {
        std::strstreambuf s;
        assert(s.str() == nullptr);
        assert(s.pcount() == 0);
    }
    {
        std::strstreambuf s(1024);
        assert(s.str() == nullptr);
        assert(s.pcount() == 0);
    }
}
