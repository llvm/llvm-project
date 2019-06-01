//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// ~queue() // implied noexcept;

// UNSUPPORTED: c++98, c++03

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

int main(int, char**)
{
    {
        typedef std::queue<MoveOnly> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }

  return 0;
}
