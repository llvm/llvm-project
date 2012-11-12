//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <queue>

// ~priority_queue() // implied noexcept;

#include <queue>
#include <cassert>

#include "../../../MoveOnly.h"

int main()
{
#if __has_feature(cxx_noexcept)
    {
        typedef std::priority_queue<MoveOnly> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }
#endif
}
