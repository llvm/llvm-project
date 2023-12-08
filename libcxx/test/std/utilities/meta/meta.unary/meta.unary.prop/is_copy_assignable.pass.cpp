//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_copy_assignable

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_copy_assignable()
{
    static_assert(( std::is_copy_assignable<T>::value), "");
#if TEST_STD_VER > 14
    static_assert(( std::is_copy_assignable_v<T>), "");
#endif
}

template <class T>
void test_is_not_copy_assignable()
{
    static_assert((!std::is_copy_assignable<T>::value), "");
#if TEST_STD_VER > 14
    static_assert((!std::is_copy_assignable_v<T>), "");
#endif
}

class Empty
{
};

class NotEmpty
{
public:
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

struct A
{
    A();
};

class B
{
    B& operator=(const B&);
};

struct C
{
    void operator=(C&);  // not const
};

int main(int, char**)
{
    test_is_copy_assignable<int> ();
    test_is_copy_assignable<int&> ();
    test_is_copy_assignable<A> ();
    test_is_copy_assignable<bit_zero> ();
    test_is_copy_assignable<Union> ();
    test_is_copy_assignable<NotEmpty> ();
    test_is_copy_assignable<Empty> ();

#if TEST_STD_VER >= 11
    test_is_not_copy_assignable<const int> ();
    test_is_not_copy_assignable<int[]> ();
    test_is_not_copy_assignable<int[3]> ();
    test_is_not_copy_assignable<B> ();
#endif
    test_is_not_copy_assignable<void> ();
    test_is_not_copy_assignable<C> ();

  return 0;
}
