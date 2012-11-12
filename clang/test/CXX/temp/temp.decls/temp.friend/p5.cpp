// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace test0 {
  template <class T> class A {
    class Member {};
  };

  class B {
    template <class T> friend class A<T>::Member;
  };

  A<int> a;
  B b;
}

// rdar://problem/8204127
namespace test1 {
  template <class T> struct A;

  class C {
    static void foo();
    template <class T> friend void A<T>::f();
  };

  template <class T> struct A {
    void f() { C::foo(); }
  };

  template <class T> struct A<T*> {
    void f() { C::foo(); }
  };

  template <> struct A<char> {
    void f() { C::foo(); }
  };
}

// FIXME: these should fail!
namespace test2 {
  template <class T> struct A;

  class C {
    static void foo();
    template <class T> friend void A<T>::g();
  };

  template <class T> struct A {
    void f() { C::foo(); }
  };

  template <class T> struct A<T*> {
    void f() { C::foo(); }
  };

  template <> struct A<char> {
    void f() { C::foo(); }
  };
}

// Tests 3, 4 and 5 were all noted in <rdar://problem/8540527>.
namespace test3 {
  template <class T> struct A {
    struct Inner {
      static int foo();
    };
  };

  template <class U> class C {
    int i;
    template <class T> friend struct A<T>::Inner;
  };

  template <class T> int A<T>::Inner::foo() {
    C<int> c;
    c.i = 0;
    return 0;
  }

  int test = A<int>::Inner::foo();
}

namespace test4 {
  template <class T> struct X {
    template <class U> void operator+=(U);
    
    template <class V>
    template <class U>
    friend void X<V>::operator+=(U);
  };

  void test() {   
    X<int>() += 1.0;
  }
}

namespace test5 {
  template<template <class> class T> struct A {
    template<template <class> class T> friend void A<T>::foo();
  };

  template <class> struct B {};
  template class A<B>;
}
