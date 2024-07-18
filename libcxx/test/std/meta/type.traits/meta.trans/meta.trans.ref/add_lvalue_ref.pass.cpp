//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// add_lvalue_reference
// If T names a referenceable type then the member typedef type
//    shall name T&; otherwise, type shall name T.

#include <type_traits>
#include "test_macros.h"

template <class T, class U>
void test_add_lvalue_reference()
{
    ASSERT_SAME_TYPE(U, typename std::add_lvalue_reference<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U, std::add_lvalue_reference_t<T>);
#endif
}

template <class F>
void test_function0()
{
    ASSERT_SAME_TYPE(F&, typename std::add_lvalue_reference<F>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(F&, std::add_lvalue_reference_t<F>);
#endif
}

template <class F>
void test_function1()
{
    ASSERT_SAME_TYPE(F, typename std::add_lvalue_reference<F>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(F, std::add_lvalue_reference_t<F>);
#endif
}

struct Foo {};

int main(int, char**)
{
    test_add_lvalue_reference<void, void>();
    test_add_lvalue_reference<int, int&>();
    test_add_lvalue_reference<int[3], int(&)[3]>();
    test_add_lvalue_reference<int&, int&>();
    test_add_lvalue_reference<const int&, const int&>();
    test_add_lvalue_reference<int*, int*&>();
    test_add_lvalue_reference<const int*, const int*&>();
    test_add_lvalue_reference<Foo, Foo&>();

//  LWG 2101 specifically talks about add_lvalue_reference and functions.
//  The term of art is "a referenceable type", which a cv- or ref-qualified function is not.
    test_function0<void()>();
    test_function1<void() const>();
    test_function1<void() &>();
    test_function1<void() &&>();
    test_function1<void() const &>();
    test_function1<void() const &&>();

//  But a cv- or ref-qualified member function *is* "a referenceable type"
    test_function0<void (Foo::*)()>();
    test_function0<void (Foo::*)() const>();
    test_function0<void (Foo::*)() &>();
    test_function0<void (Foo::*)() &&>();
    test_function0<void (Foo::*)() const &>();
    test_function0<void (Foo::*)() const &&>();

  return 0;
}
