//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX26_REMOVED_STRSTREAM

// <strstream>

// class istrstream

// char* str();

#include <strstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        const char buf[] = "123 4.5 dog";
        std::istrstream in(buf);
        assert(in.str() == std::string("123 4.5 dog"));
    }

  return 0;
}
