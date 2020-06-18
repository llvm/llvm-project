//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// ~multiset() // implied noexcept;

// UNSUPPORTED: c++03

#include <set>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

template <class T>
struct some_comp
{
    typedef T value_type;
    ~some_comp() noexcept(false);
    bool operator()(const T&, const T&) const { return false; }
};

int main(int, char**)
{
    {
        typedef std::multiset<MoveOnly> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }
    {
        typedef std::multiset<MoveOnly, std::less<MoveOnly>, test_allocator<MoveOnly>> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }
    {
        typedef std::multiset<MoveOnly, std::less<MoveOnly>, other_allocator<MoveOnly>> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }
#if defined(_LIBCPP_VERSION)
    {
        typedef std::multiset<MoveOnly, some_comp<MoveOnly>> C;
        static_assert(!std::is_nothrow_destructible<C>::value, "");
    }
#endif // _LIBCPP_VERSION

  return 0;
}
