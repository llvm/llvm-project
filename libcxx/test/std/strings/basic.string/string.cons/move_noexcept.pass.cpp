//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <string>

// basic_string(basic_string&&)
//        noexcept(is_nothrow_move_constructible<allocator_type>::value);

// This tests a conforming extension

#include <string>
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
        typedef std::string C;
        static_assert(std::is_nothrow_move_constructible<C>::value, "");
    }
    {
        typedef std::basic_string<char, std::char_traits<char>, test_allocator<char>> C;
        static_assert(std::is_nothrow_move_constructible<C>::value, "");
    }
    {
        typedef std::basic_string<char, std::char_traits<char>, some_alloc<char>> C;
#if TEST_STD_VER <= 14
        static_assert(!std::is_nothrow_move_constructible<C>::value, "");
#else
        static_assert( std::is_nothrow_move_constructible<C>::value, "");
#endif
    }
}
