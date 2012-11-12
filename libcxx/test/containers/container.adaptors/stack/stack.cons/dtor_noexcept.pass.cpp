//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <stack>

// ~stack() // implied noexcept;

#include <stack>
#include <cassert>

#include "../../../MoveOnly.h"

int main()
{
#if __has_feature(cxx_noexcept)
    {
        typedef std::stack<MoveOnly> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }
#endif
}
