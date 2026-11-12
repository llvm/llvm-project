//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// [func.require]

#include <type_traits>
#include <functional>

#include "test_macros.h"

template <typename T, int N>
struct Array
{
    typedef T type[N];
};

struct Type
{
    Array<char, 1>::type& f1();
    Array<char, 2>::type& f2() const;
};

int main(int, char**)
{
    static_assert(sizeof(std::__invoke(&Type::f1, std::declval<Type        >())) == 1, "");
    static_assert(sizeof(std::__invoke(&Type::f2, std::declval<Type const  >())) == 2, "");

  return 0;
}
