//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// check nested types:

// template <class T>
// class allocator
// {
// public:
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

int main(int, char**)
{
    static_assert((std::is_same<std::allocator<char>::value_type, char>::value), "");

    static_assert((std::is_same<std::allocator<char>::propagate_on_container_move_assignment, std::true_type>::value), "");
    LIBCPP_STATIC_ASSERT((std::is_same<std::allocator<const char>::propagate_on_container_move_assignment, std::true_type>::value), "");

    static_assert((std::is_same<std::allocator<char>::is_always_equal, std::true_type>::value), "");
    LIBCPP_STATIC_ASSERT((std::is_same<std::allocator<const char>::is_always_equal, std::true_type>::value), "");

    std::allocator<char> a;
    std::allocator<char> a2 = a;
    a2 = a;
    std::allocator<int> a3 = a2;
    ((void)a3);

  return 0;
}
