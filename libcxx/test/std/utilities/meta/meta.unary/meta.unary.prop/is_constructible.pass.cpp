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

struct Base {};
struct Derived : public Base {};

class Abstract
{
    virtual void foo() = 0;
};

class AbstractDestructor
{
    virtual ~AbstractDestructor() = 0;
};

struct PrivateDtor {
  PrivateDtor(int) {}
private:
  ~PrivateDtor() {}
};

struct S {
   template <class T>
#if TEST_STD_VER >= 11
   explicit
#endif
   operator T () const { return T(); }
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
    test_is_constructible<int, S>();
    test_is_not_constructible<int&, S>();

#if TEST_STD_VER >= 11
    test_is_constructible<int const&, int>();
    test_is_constructible<int const&, int&&>();

    test_is_not_constructible<int&, int>();
    test_is_not_constructible<int&, int const&>();
    test_is_not_constructible<int&, int&&>();

    test_is_constructible<int&&, int>();
    test_is_constructible<int&&, int&&>();
    test_is_not_constructible<int&&, int&>();
    test_is_not_constructible<int&&, int const&&>();

    test_is_constructible<Base, Derived>();
    test_is_constructible<Base&, Derived&>();
    test_is_not_constructible<Derived&, Base&>();
    test_is_constructible<Base const&, Derived const&>();
    test_is_not_constructible<Derived const&, Base const&>();
    test_is_not_constructible<Derived const&, Base>();

    test_is_constructible<Base&&, Derived>();
    test_is_constructible<Base&&, Derived&&>();
    test_is_not_constructible<Derived&&, Base&&>();
    test_is_not_constructible<Derived&&, Base>();

    // test that T must also be destructible
    test_is_constructible<PrivateDtor&, PrivateDtor&>();
    test_is_not_constructible<PrivateDtor, int>();

    test_is_not_constructible<void() const, void() const>();
    test_is_not_constructible<void() const, void*>();

// TODO: Remove this workaround once Clang <= 3.7 are no longer used regularly.
// In those compiler versions the __is_constructible builtin gives the wrong
// results for abominable function types.
#if (defined(TEST_APPLE_CLANG_VER) && TEST_APPLE_CLANG_VER < 703) \
 || (defined(TEST_CLANG_VER) && TEST_CLANG_VER < 308)
#define WORKAROUND_CLANG_BUG
#endif
#if !defined(WORKAROUND_CLANG_BUG)
    test_is_not_constructible<void()>();
    test_is_not_constructible<void() const> ();
    test_is_not_constructible<void() volatile> ();
    test_is_not_constructible<void() &> ();
    test_is_not_constructible<void() &&> ();
#endif
#endif
}
