//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <system_error>

// template <class T> constexpr bool is_error_condition_enum_v;

#include <system_error>
#include <type_traits>
#include "test_macros.h"

template <bool Expected, class T>
void
test()
{
    static_assert((std::is_error_condition_enum<T>::value == Expected), "");
#if TEST_STD_VER > 14
    static_assert((std::is_error_condition_enum_v<T>        == Expected), "");
#endif
}

class A {
    A();
    operator std::error_condition () const { return std::error_condition(); }
};

// Specialize the template for my class
namespace std
{
  template <>
  struct is_error_condition_enum<A> : public std::true_type {};
}


int main(int, char**)
{
    test<false, void>();
    test<false, int>();
    test<false, std::nullptr_t>();
    test<false, std::string>();

    test<true, A>();

  return 0;
}
