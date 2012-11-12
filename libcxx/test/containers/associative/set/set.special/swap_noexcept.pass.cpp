//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// void swap(set& c)
//     noexcept(!allocator_type::propagate_on_container_swap::value ||
//              __is_nothrow_swappable<allocator_type>::value);

// This tests a conforming extension

#include <set>
#include <cassert>

#include "../../../MoveOnly.h"
#include "../../../test_allocator.h"

template <class T>
struct some_comp
{
    typedef T value_type;
    
    some_comp() {}
    some_comp(const some_comp&) {}
    void deallocate(void*, unsigned) {}

    typedef std::true_type propagate_on_container_swap;
};

int main()
{
#if __has_feature(cxx_noexcept)
    {
        typedef std::set<MoveOnly> C;
        C c1, c2;
        static_assert(noexcept(swap(c1, c2)), "");
    }
    {
        typedef std::set<MoveOnly, std::less<MoveOnly>, test_allocator<MoveOnly>> C;
        C c1, c2;
        static_assert(noexcept(swap(c1, c2)), "");
    }
    {
        typedef std::set<MoveOnly, std::less<MoveOnly>, other_allocator<MoveOnly>> C;
        C c1, c2;
        static_assert(noexcept(swap(c1, c2)), "");
    }
    {
        typedef std::set<MoveOnly, some_comp<MoveOnly>> C;
        C c1, c2;
        static_assert(!noexcept(swap(c1, c2)), "");
    }
#endif
}
