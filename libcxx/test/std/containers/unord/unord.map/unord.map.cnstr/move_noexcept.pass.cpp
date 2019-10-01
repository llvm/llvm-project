//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// unordered_map(unordered_map&&)
//        noexcept(is_nothrow_move_constructible<allocator_type>::value &&
//                 is_nothrow_move_constructible<key_compare>::value);

// This tests a conforming extension

// UNSUPPORTED: c++98, c++03

#include <unordered_map>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

template <class T>
struct some_comp
{
    typedef T value_type;
    some_comp(const some_comp&);
    bool operator()(const T&, const T&) const { return false; }
};

template <class T>
struct some_hash
{
    typedef T value_type;
    some_hash();
    some_hash(const some_hash&);

    std::size_t operator()(T const&) const;
};

int main(int, char**)
{
#if defined(_LIBCPP_VERSION)
    {
        typedef std::unordered_map<MoveOnly, MoveOnly> C;
        static_assert(std::is_nothrow_move_constructible<C>::value, "");
    }
    {
        typedef std::unordered_map<MoveOnly, MoveOnly, std::hash<MoveOnly>,
                           std::equal_to<MoveOnly>, test_allocator<std::pair<const MoveOnly, MoveOnly>>> C;
        static_assert(std::is_nothrow_move_constructible<C>::value, "");
    }
    {
        typedef std::unordered_map<MoveOnly, MoveOnly, std::hash<MoveOnly>,
                          std::equal_to<MoveOnly>, other_allocator<std::pair<const MoveOnly, MoveOnly>>> C;
        static_assert(std::is_nothrow_move_constructible<C>::value, "");
    }
#endif // _LIBCPP_VERSION
    {
        typedef std::unordered_map<MoveOnly, MoveOnly, some_hash<MoveOnly>> C;
        static_assert(!std::is_nothrow_move_constructible<C>::value, "");
    }
    {
        typedef std::unordered_map<MoveOnly, MoveOnly, std::hash<MoveOnly>,
                                                         some_comp<MoveOnly>> C;
        static_assert(!std::is_nothrow_move_constructible<C>::value, "");
    }

  return 0;
}
