//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03

// <system_error>

// template <> struct is_error_code_enum<> : public false_type {};

#include <system_error>
#include <string>
#include "test_macros.h"

template <bool Expected, class T>
void
test()
{
    static_assert((std::is_error_code_enum<T>::value == Expected), "");
#if TEST_STD_VER > 14
    static_assert((std::is_error_code_enum_v<T>      == Expected), "");
#endif
}

class A {
    A();
    operator std::error_code () const { return std::error_code(); }
};

// Specialize the template for my class
namespace std
{
  template <>
  struct is_error_code_enum<A> : public std::true_type {};
}


int main()
{
    test<false, void>();
    test<false, int>();
    test<false, std::nullptr_t>();
    test<false, std::string>();

    test<true, A>();
}
