//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstreambuf

// explicit strstreambuf(streamsize alsize_arg = 0); // before C++20
// strstreambuf() : strstreambuf(0) {}               // C++20
// explicit strstreambuf(streamsize alsize_arg);     // C++20

#include <strstream>
#include <cassert>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "test_convertible.h"
#endif

int main(int, char**)
{
    {
        std::strstreambuf s;
        assert(s.str() == nullptr);
        assert(s.pcount() == 0);
    }
    {
        std::strstreambuf s(1024);
        LIBCPP_ASSERT(s.str() == nullptr);
        assert(s.pcount() == 0);
    }

#if TEST_STD_VER >= 11
    {
      typedef std::strstreambuf B;
      static_assert(test_convertible<B>(), "");
      static_assert(!test_convertible<B, std::streamsize>(), "");
    }
#endif

    return 0;
}
