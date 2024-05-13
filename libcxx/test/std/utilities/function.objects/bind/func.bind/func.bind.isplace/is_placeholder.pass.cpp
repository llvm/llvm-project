//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// struct is_placeholder

#include <functional>
#include "test_macros.h"

template <int Expected, class T>
void
test(const T&)
{
    static_assert(std::is_placeholder<T>::value == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_placeholder<T&>::value == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_placeholder<const T>::value == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_placeholder<const T&>::value == Expected, "");
    static_assert(std::is_base_of<std::integral_constant<int, Expected>, std::is_placeholder<T> >::value, "");
    LIBCPP_STATIC_ASSERT(std::is_base_of<std::integral_constant<int, Expected>, std::is_placeholder<T&> >::value, "");
    LIBCPP_STATIC_ASSERT(std::is_base_of<std::integral_constant<int, Expected>, std::is_placeholder<const T> >::value, "");
    LIBCPP_STATIC_ASSERT(std::is_base_of<std::integral_constant<int, Expected>, std::is_placeholder<const T&> >::value, "");

#if TEST_STD_VER > 14
    ASSERT_SAME_TYPE(decltype(std::is_placeholder_v<T>), const int);
    static_assert(std::is_placeholder_v<T> == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_placeholder_v<T&> == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_placeholder_v<const T> == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_placeholder_v<const T&> == Expected, "");
#endif
}

struct C {};

int main(int, char**)
{
    test<1>(std::placeholders::_1);
    test<2>(std::placeholders::_2);
    test<3>(std::placeholders::_3);
    test<4>(std::placeholders::_4);
    test<5>(std::placeholders::_5);
    test<6>(std::placeholders::_6);
    test<7>(std::placeholders::_7);
    test<8>(std::placeholders::_8);
    test<9>(std::placeholders::_9);
    test<10>(std::placeholders::_10);
    test<0>(4);
    test<0>(5.5);
    test<0>('a');
    test<0>(C());

  return 0;
}
