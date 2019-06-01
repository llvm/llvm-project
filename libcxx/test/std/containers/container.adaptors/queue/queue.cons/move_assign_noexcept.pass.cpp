//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// queue& operator=(queue&& c)
//     noexcept(is_nothrow_move_assignable<container_type>::value);

// This tests a conforming extension

// UNSUPPORTED: c++98, c++03

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

int main(int, char**)
{
    {
        typedef std::queue<MoveOnly> C;
        static_assert(std::is_nothrow_move_assignable<C>::value, "");
    }

  return 0;
}
