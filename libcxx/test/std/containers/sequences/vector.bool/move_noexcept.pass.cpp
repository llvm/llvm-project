//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// vector(vector&&)
//        noexcept(is_nothrow_move_constructible<allocator_type>::value);

// This tests a conforming extension

// UNSUPPORTED: c++98, c++03

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

template <class T>
struct some_alloc
{
    typedef T value_type;
    some_alloc(const some_alloc&);
};

int main()
{
    {
        typedef std::vector<bool> C;
        LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible<C>::value, "");
    }
    {
        typedef std::vector<bool, test_allocator<bool>> C;
        LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible<C>::value, "");
    }
    {
        typedef std::vector<bool, other_allocator<bool>> C;
        LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible<C>::value, "");
    }
    {
        typedef std::vector<bool, some_alloc<bool>> C;
    //  In C++17, move constructors for allocators are not allowed to throw
#if TEST_STD_VER > 14
        LIBCPP_STATIC_ASSERT( std::is_nothrow_move_constructible<C>::value, "");
#else
        static_assert(!std::is_nothrow_move_constructible<C>::value, "");
#endif
    }
}
