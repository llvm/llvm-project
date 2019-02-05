//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T> typename add_rvalue_reference<T>::type declval() noexcept;

#include <utility>
#include <type_traits>

#include "test_macros.h"

class A
{
    A(const A&);
    A& operator=(const A&);
};

int main(int, char**)
{
#if TEST_STD_VER >= 11
    static_assert((std::is_same<decltype(std::declval<A>()), A&&>::value), "");
#else
    static_assert((std::is_same<decltype(std::declval<A>()), A&>::value), "");
#endif

  return 0;
}
