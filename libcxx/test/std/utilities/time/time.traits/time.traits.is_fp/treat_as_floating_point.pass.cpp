//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// treat_as_floating_point

#include <chrono>
#include <type_traits>

#include "test_macros.h"

template <class T>
void
test()
{
    static_assert((std::is_base_of<std::is_floating_point<T>,
                                   std::chrono::treat_as_floating_point<T> >::value), "");
#if TEST_STD_VER > 14
    static_assert(std::is_floating_point<T>::value ==
                                  std::chrono::treat_as_floating_point_v<T>, "");
#endif
}

struct A {};

int main()
{
    test<int>();
    test<unsigned>();
    test<char>();
    test<bool>();
    test<float>();
    test<double>();
    test<long double>();
    test<A>();
}
