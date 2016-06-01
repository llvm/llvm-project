//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <map>

// void swap(multimap& c)
//     noexcept(!allocator_type::propagate_on_container_swap::value ||
//              __is_nothrow_swappable<allocator_type>::value);
//
//  In C++17, the standard says that swap shall have:
//     noexcept(allocator_traits<Allocator>::is_always_equal::value &&
//              noexcept(swap(declval<Compare&>(), declval<Compare&>())));

// This tests a conforming extension

#include <map>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

template <class T>
struct some_comp
{
    typedef T value_type;

    some_comp() {}
    some_comp(const some_comp&) {}
    void deallocate(void*, unsigned) {}

    typedef std::true_type propagate_on_container_swap;
};

template <class T>
struct some_comp2
{
    typedef T value_type;

    some_comp2() {}
    some_comp2(const some_comp2&) {}
    void deallocate(void*, unsigned) {}
    typedef std::true_type propagate_on_container_swap;
};

#if TEST_STD_VER >= 14
template <typename T>
void swap(some_comp2<T>&, some_comp2<T>&) noexcept {}
#endif

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

template <class T>
struct some_alloc3
{
    typedef T value_type;

    some_alloc3() {}
    some_alloc3(const some_alloc3&);
    void deallocate(void*, unsigned) {}

    typedef std::false_type propagate_on_container_swap;
    typedef std::false_type is_always_equal;
};

int main()
{
    typedef std::pair<const MoveOnly, MoveOnly> V;
    {
        typedef std::multimap<MoveOnly, MoveOnly> C;
        C c1, c2;
        static_assert(noexcept(swap(c1, c2)), "");
    }
    {
        typedef std::multimap<MoveOnly, MoveOnly, std::less<MoveOnly>, test_allocator<V>> C;
        C c1, c2;
        static_assert(noexcept(swap(c1, c2)), "");
    }
    {
        typedef std::multimap<MoveOnly, MoveOnly, std::less<MoveOnly>, other_allocator<V>> C;
        C c1, c2;
        static_assert(noexcept(swap(c1, c2)), "");
    }
    {
        typedef std::multimap<MoveOnly, MoveOnly, some_comp<MoveOnly>> C;
        C c1, c2;
        static_assert(!noexcept(swap(c1, c2)), "");
    }

#if TEST_STD_VER >= 14
    { // POCS allocator, throwable swap for comp
    typedef std::multimap<MoveOnly, MoveOnly, some_comp <MoveOnly>, some_alloc <V>> C;
    C c1, c2;
    static_assert(!noexcept(swap(c1, c2)), "");
    }
    { // always equal allocator, throwable swap for comp
    typedef std::multimap<MoveOnly, MoveOnly, some_comp <MoveOnly>, some_alloc2<V>> C;
    C c1, c2;
    static_assert(!noexcept(swap(c1, c2)), "");
    }
    { // POCS allocator, nothrow swap for comp
    typedef std::multimap<MoveOnly, MoveOnly, some_comp2<MoveOnly>, some_alloc <V>> C;
    C c1, c2;
    static_assert( noexcept(swap(c1, c2)), "");
    }
    { // always equal allocator, nothrow swap for comp
    typedef std::multimap<MoveOnly, MoveOnly, some_comp2<MoveOnly>, some_alloc2<V>> C;
    C c1, c2;
    static_assert( noexcept(swap(c1, c2)), "");
    }

    { // NOT always equal allocator, nothrow swap for comp
    typedef std::map<MoveOnly, MoveOnly, some_comp2<MoveOnly>, some_alloc3<V>> C;
    C c1, c2;
    static_assert( noexcept(swap(c1, c2)), "");
    }
#endif
}
