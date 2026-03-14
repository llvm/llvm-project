//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX26_REMOVED_STRSTREAM

// <strstream>

// class strstream

// void freeze(bool freezefl = true);

#include <strstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::strstream out;
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

  return 0;
}
