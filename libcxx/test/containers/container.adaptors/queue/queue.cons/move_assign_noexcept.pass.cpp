//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <queue>

// queue& operator=(queue&& c)
//     noexcept(is_nothrow_move_assignable<container_type>::value);

// This tests a conforming extension

#include <queue>
#include <cassert>

#include "../../../MoveOnly.h"

int main()
{
#if __has_feature(cxx_noexcept)
    {
        typedef std::queue<MoveOnly> C;
        static_assert(std::is_nothrow_move_assignable<C>::value, "");
    }
#endif
}
