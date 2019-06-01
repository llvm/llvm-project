//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Ptr>
// struct pointer_traits
// {
//     typedef Ptr pointer;
//     ...
// };

#include <memory>
#include <type_traits>

#include "test_macros.h"

struct A
{
    typedef short element_type;
    typedef char difference_type;
};

int main(int, char**)
{
    static_assert((std::is_same<std::pointer_traits<A>::pointer, A>::value), "");
    static_assert((std::is_same<std::pointer_traits<int*>::pointer, int*>::value), "");

  return 0;
}
