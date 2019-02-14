//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <string>

// basic_string& operator=(basic_string&& c)
//     noexcept(
//         allocator_traits<allocator_type>::propagate_on_container_move_assignment::value ||
//         allocator_traits<allocator_type>::is_always_equal::value); // C++17
//
//  before C++17, we use the conforming extension
//     noexcept(
//         allocator_type::propagate_on_container_move_assignment::value &&
//         is_nothrow_move_assignable<allocator_type>::value);

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

template <class T>
struct some_alloc
{
    typedef T value_type;
    some_alloc(const some_alloc&);
    T *allocate(size_t);
};

template <class T>
struct some_alloc2
{
    typedef T value_type;

    some_alloc2() {}
    some_alloc2(const some_alloc2&);
    T *allocate(size_t);
    void deallocate(void*, unsigned) {}

    typedef std::false_type propagate_on_container_move_assignment;
    typedef std::true_type is_always_equal;
};

template <class T>
struct some_alloc3
{
    typedef T value_type;

    some_alloc3() {}
    some_alloc3(const some_alloc3&);
    T *allocate(size_t);
    void deallocate(void*, unsigned) {}

    typedef std::false_type propagate_on_container_move_assignment;
    typedef std::false_type is_always_equal;
};

int main(int, char**)
{
    {
        typedef std::string C;
        static_assert(std::is_nothrow_move_assignable<C>::value, "");
    }
    {
        typedef std::basic_string<char, std::char_traits<char>, test_allocator<char>> C;
        static_assert(!std::is_nothrow_move_assignable<C>::value, "");
    }
    {
        typedef std::basic_string<char, std::char_traits<char>, some_alloc<char>> C;
#if TEST_STD_VER > 14
    //  if the allocators are always equal, then the move assignment can be noexcept
        static_assert( std::is_nothrow_move_assignable<C>::value, "");
#else
        static_assert(!std::is_nothrow_move_assignable<C>::value, "");
#endif
    }
#if TEST_STD_VER > 14
    {
    //  POCMA is false, always equal
        typedef std::basic_string<char, std::char_traits<char>, some_alloc2<char>> C;
        static_assert( std::is_nothrow_move_assignable<C>::value, "");
    }
    {
    //  POCMA is false, not always equal
        typedef std::basic_string<char, std::char_traits<char>, some_alloc3<char>> C;
        static_assert(!std::is_nothrow_move_assignable<C>::value, "");
    }
#endif

  return 0;
}
