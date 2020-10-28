//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Check that the nested types of std::allocator are provided:

// template <class T>
// class allocator
// {
// public:
//     typedef size_t    size_type;
//     typedef ptrdiff_t difference_type;
//     typedef T         value_type;
//
//     typedef true_type propagate_on_container_move_assignment;
//     typedef true_type is_always_equal;
// ...
// };

#include <memory>
#include <type_traits>
#include <cstddef>

#include "test_macros.h"

template <typename T, typename U>
TEST_CONSTEXPR_CXX20 bool test()
{
    static_assert((std::is_same<typename std::allocator<T>::size_type, std::size_t>::value), "");
    static_assert((std::is_same<typename std::allocator<T>::difference_type, std::ptrdiff_t>::value), "");
    static_assert((std::is_same<typename std::allocator<T>::value_type, T>::value), "");
    static_assert((std::is_same<typename std::allocator<T>::propagate_on_container_move_assignment, std::true_type>::value), "");
    static_assert((std::is_same<typename std::allocator<T>::is_always_equal, std::true_type>::value), "");

    std::allocator<T> a;
    std::allocator<T> a2 = a;
    a2 = a;
    std::allocator<U> a3 = a2;
    (void)a3;

    return true;
}

int main(int, char**)
{
    test<char, int>();
    test<char const, int const>();
#if TEST_STD_VER > 17
    static_assert(test<char, int>());
    static_assert(test<char const, int const>());
#endif
    return 0;
}
