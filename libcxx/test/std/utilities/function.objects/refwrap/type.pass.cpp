//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// check for member typedef type

#include <functional>
#include <type_traits>

class C {};

int main(int, char**)
{
    static_assert((std::is_same<std::reference_wrapper<C>::type,
                                                       C>::value), "");
    static_assert((std::is_same<std::reference_wrapper<void ()>::type,
                                                       void ()>::value), "");
    static_assert((std::is_same<std::reference_wrapper<int* (double*)>::type,
                                                       int* (double*)>::value), "");
    static_assert((std::is_same<std::reference_wrapper<void(*)()>::type,
                                                       void(*)()>::value), "");
    static_assert((std::is_same<std::reference_wrapper<int*(*)(double*)>::type,
                                                       int*(*)(double*)>::value), "");
    static_assert((std::is_same<std::reference_wrapper<int*(C::*)(double*)>::type,
                                                       int*(C::*)(double*)>::value), "");
    static_assert((std::is_same<std::reference_wrapper<int (C::*)(double*) const volatile>::type,
                                                       int (C::*)(double*) const volatile>::value), "");

  return 0;
}
