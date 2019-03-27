//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// decay

#include <type_traits>

#include "test_macros.h"

template <class T, class U>
void test_decay()
{
    static_assert((std::is_same<typename std::decay<T>::type, U>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::decay_t<T>,     U>::value), "");
#endif
}

int main(int, char**)
{
    test_decay<void, void>();
    test_decay<int, int>();
    test_decay<const volatile int, int>();
    test_decay<int*, int*>();
    test_decay<int[3], int*>();
    test_decay<const int[3], const int*>();
    test_decay<void(), void (*)()>();
#if TEST_STD_VER > 11
    test_decay<int(int) const, int(int) const>();
    test_decay<int(int) volatile, int(int) volatile>();
    test_decay<int(int)  &, int(int)  &>();
    test_decay<int(int) &&, int(int) &&>();
#endif

  return 0;
}
