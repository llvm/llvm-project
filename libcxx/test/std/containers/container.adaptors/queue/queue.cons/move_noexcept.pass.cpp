//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <queue>

// queue(queue&&)
//        noexcept(is_nothrow_move_constructible<container_type>::value);

// This tests a conforming extension

// UNSUPPORTED: c++98, c++03

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

int main()
{
    {
        typedef std::queue<MoveOnly> C;
        LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible<C>::value, "");
    }
}
