//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostreambuf_iterator

// bool failed() const throw();
//
// Extension: constructing from NULL is UB; we just make it a failed iterator

#include <iterator>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::ostreambuf_iterator<char> i(nullptr);
        assert(i.failed());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::ostreambuf_iterator<wchar_t> i(nullptr);
        assert(i.failed());
    }
#endif

  return 0;
}
