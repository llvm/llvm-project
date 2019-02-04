//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// void swap(stack& c)
//     noexcept(__is_nothrow_swappable<container_type>::value);

// This tests a conforming extension

// UNSUPPORTED: c++98, c++03

#include <stack>
#include <utility>
#include <cassert>

#include "MoveOnly.h"

int main(int, char**)
{
    {
        typedef std::stack<MoveOnly> C;
        static_assert(noexcept(swap(std::declval<C&>(), std::declval<C&>())), "");
    }

  return 0;
}
