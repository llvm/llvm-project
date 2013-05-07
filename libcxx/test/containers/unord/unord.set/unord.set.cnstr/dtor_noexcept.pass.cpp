//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// ~unordered_set() // implied noexcept;

#include <unordered_set>
#include <cassert>

#include "../../../MoveOnly.h"
#include "../../../test_allocator.h"

#if __has_feature(cxx_noexcept)

template <class T>
struct some_comp
{
    typedef T value_type;
    ~some_comp() noexcept(false);
};

template <class T>
struct some_hash
{
    typedef T value_type;
    some_hash();
    some_hash(const some_hash&);
    ~some_hash() noexcept(false);
};

#endif

int main()
{
#if __has_feature(cxx_noexcept)
    {
        typedef std::unordered_set<MoveOnly> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }
    {
        typedef std::unordered_set<MoveOnly, std::hash<MoveOnly>,
                           std::equal_to<MoveOnly>, test_allocator<MoveOnly>> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }
    {
        typedef std::unordered_set<MoveOnly, std::hash<MoveOnly>,
                          std::equal_to<MoveOnly>, other_allocator<MoveOnly>> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }
    {
        typedef std::unordered_set<MoveOnly, some_hash<MoveOnly>> C;
        static_assert(!std::is_nothrow_destructible<C>::value, "");
    }
    {
        typedef std::unordered_set<MoveOnly, std::hash<MoveOnly>,
                                                         some_comp<MoveOnly>> C;
        static_assert(!std::is_nothrow_destructible<C>::value, "");
    }
#endif
}
