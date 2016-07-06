//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits
// XFAIL: apple-clang-6.0
//	The Apple-6 compiler gets is_constructible<void ()> wrong.

// template <class T, class... Args>
//   struct is_constructible;

#include <type_traits>
#include "test_macros.h"

struct A
{
    explicit A(int);
    A(int, double);
#if TEST_STD_VER >= 11
private:
#endif
    A(char);
};

class Abstract
{
    virtual void foo() = 0;
};

class AbstractDestructor
{
    virtual ~AbstractDestructor() = 0;
};

template <class T>
void test_is_constructible()
{
    static_assert( (std::is_constructible<T>::value), "");
#if TEST_STD_VER > 14
    static_assert( std::is_constructible_v<T>, "");
#endif
}

template <class T, class A0>
void test_is_constructible()
{
    static_assert(( std::is_constructible<T, A0>::value), "");
#if TEST_STD_VER > 14
    static_assert(( std::is_constructible_v<T, A0>), "");
#endif
}

template <class T, class A0, class A1>
void test_is_constructible()
{
    static_assert(( std::is_constructible<T, A0, A1>::value), "");
#if TEST_STD_VER > 14
    static_assert(( std::is_constructible_v<T, A0, A1>), "");
#endif
}

template <class T>
void test_is_not_constructible()
{
    static_assert((!std::is_constructible<T>::value), "");
#if TEST_STD_VER > 14
    static_assert((!std::is_constructible_v<T>), "");
#endif
}

template <class T, class A0>
void test_is_not_constructible()
{
    static_assert((!std::is_constructible<T, A0>::value), "");
#if TEST_STD_VER > 14
    static_assert((!std::is_constructible_v<T, A0>), "");
#endif
}

int main()
{
    test_is_constructible<int> ();
    test_is_constructible<int, const int> ();
    test_is_constructible<A, int> ();
    test_is_constructible<A, int, double> ();
    test_is_constructible<int&, int&> ();

    test_is_not_constructible<A> ();
#if TEST_STD_VER >= 11
    test_is_not_constructible<A, char> ();
#else
    test_is_constructible<A, char> ();
#endif
    test_is_not_constructible<A, void> ();
    test_is_not_constructible<void> ();
    test_is_not_constructible<int&> ();
    test_is_not_constructible<Abstract> ();
    test_is_not_constructible<AbstractDestructor> ();

//  LWG 2560  -- postpone this test until bots updated
    test_is_not_constructible<void()> ();
#if TEST_STD_VER > 11
    test_is_not_constructible<void() const> ();
    test_is_not_constructible<void() volatile> ();
    test_is_not_constructible<void() &> ();
    test_is_not_constructible<void() &&> ();
#endif
}
