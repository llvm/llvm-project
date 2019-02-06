//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T>
// struct pointer_traits<T*>
// {
//     template <class U> using rebind = U*;
//     ...
// };

#include <memory>
#include <type_traits>

#include "test_macros.h"

int main()
{
#if TEST_STD_VER >= 11
    static_assert((std::is_same<std::pointer_traits<int*>::rebind<double>, double*>::value), "");
#else
    static_assert((std::is_same<std::pointer_traits<int*>::rebind<double>::other, double*>::value), "");
#endif
}
