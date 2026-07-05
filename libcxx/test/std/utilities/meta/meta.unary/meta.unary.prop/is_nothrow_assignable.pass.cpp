//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_assignable

#include <type_traits>
#include "test_macros.h"

template <class T, class U>
void test_is_nothrow_assignable()
{
    static_assert(( std::is_nothrow_assignable<T, U>::value), "");
#if TEST_STD_VER > 14
    static_assert(( std::is_nothrow_assignable_v<T, U>), "");
#endif
}

template <class T, class U>
void test_is_not_nothrow_assignable()
{
    static_assert((!std::is_nothrow_assignable<T, U>::value), "");
#if TEST_STD_VER > 14
    static_assert((!std::is_nothrow_assignable_v<T, U>), "");
#endif
}

struct A
{
};

struct B
{
    void operator=(A);
};

struct C
{
    void operator=(C&);  // not const
};

int main(int, char**)
{
    test_is_nothrow_assignable<int&, int&> ();
    test_is_nothrow_assignable<int&, int> ();
#if TEST_STD_VER >= 11
    test_is_nothrow_assignable<int&, double> ();
#endif

    test_is_not_nothrow_assignable<int, int&> ();
    test_is_not_nothrow_assignable<int, int> ();
    test_is_not_nothrow_assignable<B, A> ();
    test_is_not_nothrow_assignable<A, B> ();
    test_is_not_nothrow_assignable<C, C&> ();

  return 0;
}
