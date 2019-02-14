//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// class array

// bool empty() const noexcept;

#include <array>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef std::array<int, 2> C;
    C c;
    ASSERT_NOEXCEPT(c.empty());
    assert(!c.empty());
    }
    {
    typedef std::array<int, 0> C;
    C c;
    ASSERT_NOEXCEPT(c.empty());
    assert( c.empty());
    }

  return 0;
}
