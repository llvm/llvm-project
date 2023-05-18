//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// template <class T, class... Args>
//   struct is_constructible;

#include <type_traits>
#include "test_macros.h"

struct A
{
    explicit A(int);
    A(int, double);
    A(int, long, double);
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
   operator T () const;
};

template <class To>
struct ImplicitTo {
  operator To();
};

#if TEST_STD_VER >= 11
template <class To>
struct ExplicitTo {
   explicit operator To ();
};
#endif


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

template <class T, class A0, class A1, class A2>
void test_is_constructible()
{
    static_assert(( std::is_constructible<T, A0, A1, A2>::value), "");
#if TEST_STD_VER > 14
    static_assert(( std::is_constructible_v<T, A0, A1, A2>), "");
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

int main(int, char**)
{
    typedef Base B;
    typedef Derived D;

    test_is_constructible<int> ();
    test_is_constructible<int, const int> ();
    test_is_constructible<A, int> ();
    test_is_constructible<A, int, double> ();
    test_is_constructible<A, int, long, double> ();
    test_is_constructible<int&, int&> ();

    test_is_not_constructible<A> ();
#if TEST_STD_VER >= 11
    test_is_not_constructible<A, char> ();
#else
    test_is_constructible<A, char> ();
#endif
    test_is_not_constructible<A, void> ();
    test_is_not_constructible<int, void()>();
    test_is_not_constructible<int, void(&)()>();
    test_is_not_constructible<int, void() const>();
    test_is_not_constructible<int&, void>();
    test_is_not_constructible<int&, void()>();
    test_is_not_constructible<int&, void() const>();
    test_is_not_constructible<int&, void(&)()>();

    test_is_not_constructible<void> ();
    test_is_not_constructible<const void> ();  // LWG 2738
    test_is_not_constructible<volatile void> ();
    test_is_not_constructible<const volatile void> ();
    test_is_not_constructible<int&> ();
    test_is_not_constructible<Abstract> ();
    test_is_not_constructible<AbstractDestructor> ();
    test_is_constructible<int, S>();
    test_is_not_constructible<int&, S>();

    test_is_constructible<void(&)(), void(&)()>();
    test_is_constructible<void(&)(), void()>();
#if TEST_STD_VER >= 11
    test_is_constructible<void(&&)(), void(&&)()>();
    test_is_constructible<void(&&)(), void()>();
    test_is_constructible<void(&&)(), void(&)()>();
#endif

#if TEST_STD_VER >= 11
    test_is_constructible<int const&, int>();
    test_is_constructible<int const&, int&&>();

    test_is_constructible<int&&, double&>();
    test_is_constructible<void(&)(), void(&&)()>();

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
#ifndef TEST_COMPILER_GCC
    test_is_not_constructible<Derived const&, Base const&>();
    test_is_not_constructible<Derived const&, Base>();
#endif

    test_is_constructible<Base&&, Derived>();
    test_is_constructible<Base&&, Derived&&>();
#ifndef TEST_COMPILER_GCC
    test_is_not_constructible<Derived&&, Base&&>();
    test_is_not_constructible<Derived&&, Base>();
#endif

    // test that T must also be destructible
    test_is_constructible<PrivateDtor&, PrivateDtor&>();
    test_is_not_constructible<PrivateDtor, int>();

    test_is_not_constructible<void() const, void() const>();
    test_is_not_constructible<void() const, void*>();

    test_is_constructible<int&, ImplicitTo<int&>>();
    test_is_constructible<const int&, ImplicitTo<int&&>>();
    test_is_constructible<int&&, ImplicitTo<int&&>>();
    test_is_constructible<const int&, ImplicitTo<int>>();

    test_is_not_constructible<B&&, B&>();
    test_is_not_constructible<B&&, D&>();
    test_is_constructible<B&&, ImplicitTo<D&&>>();
    test_is_constructible<B&&, ImplicitTo<D&&>&>();
    test_is_constructible<int&&, double&>();
    test_is_constructible<const int&, ImplicitTo<int&>&>();
    test_is_constructible<const int&, ImplicitTo<int&>>();
    test_is_constructible<const int&, ExplicitTo<int&>&>();
    test_is_constructible<const int&, ExplicitTo<int&>>();

    test_is_constructible<const int&, ExplicitTo<int&>&>();
    test_is_constructible<const int&, ExplicitTo<int&>>();


    // Binding through reference-compatible type is required to perform
    // direct-initialization as described in [over.match.ref] p. 1 b. 1:
    //
    // But the rvalue to lvalue reference binding isn't allowed according to
    // [over.match.ref] despite Clang accepting it.
    test_is_constructible<int&, ExplicitTo<int&>>();
#ifndef TEST_COMPILER_GCC
    test_is_constructible<const int&, ExplicitTo<int&&>>();
#endif

    static_assert(std::is_constructible<int&&, ExplicitTo<int&&>>::value, "");

#ifdef __clang__
    // FIXME Clang and GCC disagree on the validity of this expression.
    test_is_constructible<const int&, ExplicitTo<int>>();
    static_assert(std::is_constructible<int&&, ExplicitTo<int>>::value, "");
#else
    test_is_not_constructible<const int&, ExplicitTo<int>>();
    test_is_not_constructible<int&&, ExplicitTo<int>>();
#endif

    // Binding through temporary behaves like copy-initialization,
    // see [dcl.init.ref] p. 5, very last sub-bullet:
    test_is_not_constructible<const int&, ExplicitTo<double&&>>();
    test_is_not_constructible<int&&, ExplicitTo<double&&>>();

    test_is_not_constructible<void()>();
    test_is_not_constructible<void() const> ();
    test_is_not_constructible<void() volatile> ();
    test_is_not_constructible<void() &> ();
    test_is_not_constructible<void() &&> ();
#endif // TEST_STD_VER >= 11

  return 0;
}
