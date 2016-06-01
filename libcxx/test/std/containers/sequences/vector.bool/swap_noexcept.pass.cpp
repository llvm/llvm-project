//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <vector>

// void swap(vector& c)
//     noexcept(!allocator_type::propagate_on_container_swap::value ||
//              __is_nothrow_swappable<allocator_type>::value);
//
//  In C++17, the standard says that swap shall have:
//     noexcept(allocator_traits<Allocator>::propagate_on_container_swap::value ||
//              allocator_traits<Allocator>::is_always_equal::value);

// This tests a conforming extension

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

template <class T>
struct some_alloc
{
    typedef T value_type;

    some_alloc() {}
    some_alloc(const some_alloc&);
    void deallocate(void*, unsigned) {}

    typedef std::true_type propagate_on_container_swap;
};

template <class T>
struct some_alloc2
{
    typedef T value_type;

    some_alloc2() {}
    some_alloc2(const some_alloc2&);
    void deallocate(void*, unsigned) {}

    typedef std::false_type propagate_on_container_swap;
    typedef std::true_type is_always_equal;
};

int main()
{
    {
        typedef std::vector<bool> C;
        C c1, c2;
        static_assert(noexcept(swap(c1, c2)), "");
    }
    {
        typedef std::vector<bool, test_allocator<bool>> C;
        C c1, c2;
        static_assert(noexcept(swap(c1, c2)), "");
    }
    {
        typedef std::vector<bool, other_allocator<bool>> C;
        C c1, c2;
        static_assert(noexcept(swap(c1, c2)), "");
    }
    {
        typedef std::vector<bool, some_alloc<bool>> C;
        C c1, c2;
#if TEST_STD_VER >= 14
    //  In c++14, if POCS is set, swapping the allocator is required not to throw
        static_assert( noexcept(swap(c1, c2)), "");
#else
        static_assert(!noexcept(swap(c1, c2)), "");
#endif
    }
#if TEST_STD_VER >= 14
    {
        typedef std::vector<bool, some_alloc2<bool>> C;
        C c1, c2;
    //  if the allocators are always equal, then the swap can be noexcept
        static_assert( noexcept(swap(c1, c2)), "");
    }
#endif
}
