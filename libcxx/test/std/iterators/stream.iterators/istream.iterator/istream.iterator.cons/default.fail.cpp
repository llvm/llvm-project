//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// constexpr istream_iterator();

#include <iterator>
#include <cassert>

#include "test_macros.h"

struct S { S(); }; // not constexpr

int main()
{
#if TEST_STD_VER >= 11
    {
    constexpr std::istream_iterator<S> it;
    }
#else
#error "C++11 only test"
#endif
}
