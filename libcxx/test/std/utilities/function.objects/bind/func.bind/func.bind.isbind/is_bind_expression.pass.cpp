//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// template<class T> struct is_bind_expression

#include <functional>
#include <type_traits>

#include "test_macros.h"

template <bool Expected, class T>
void
test(const T&)
{
    static_assert(std::is_bind_expression<T>::value == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_bind_expression<T&>::value == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_bind_expression<const T>::value == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_bind_expression<const T&>::value == Expected, "");
    static_assert(std::is_base_of<std::integral_constant<bool, Expected>, std::is_bind_expression<T> >::value, "");

#if TEST_STD_VER > 14
    ASSERT_SAME_TYPE(decltype(std::is_bind_expression_v<T>), const bool);
    static_assert(std::is_bind_expression_v<T> == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_bind_expression_v<T&> == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_bind_expression_v<const T> == Expected, "");
    LIBCPP_STATIC_ASSERT(std::is_bind_expression_v<const T&> == Expected, "");
#endif
}

struct C {int operator()(...) const { return 0; }};

int main(int, char**)
{
    test<true>(std::bind(C()));
    test<true>(std::bind(C(), std::placeholders::_2));
    test<true>(std::bind<int>(C()));
    test<false>(1);
    test<false>(std::placeholders::_2);

  return 0;
}
