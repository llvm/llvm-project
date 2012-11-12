//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class... Args> void emplace(Args&&... args);

#include <queue>
#include <cassert>

#include "../../../Emplaceable.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::queue<Emplaceable> q;
    q.emplace(1, 2.5);
    q.emplace(2, 3.5);
    q.emplace(3, 4.5);
    assert(q.size() == 3);
    assert(q.front() == Emplaceable(1, 2.5));
    assert(q.back() == Emplaceable(3, 4.5));
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
