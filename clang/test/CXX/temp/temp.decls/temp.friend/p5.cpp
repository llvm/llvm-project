// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  template <class T> class A {
    class Member {};
  };

  class B {
    template <class T> friend class A<T>::Member; // expected-warning {{not supported}}
    int n;
  };

  A<int> a;
  B b;
}

// rdar://problem/8204127
namespace test1 {
  template <class T> struct A;

  class C {
    static void foo();
    template <class T> friend void A<T>::f(); // expected-warning {{not supported}}
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
    template <class T> friend void A<T>::g(); // expected-warning {{not supported}}
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
    template <class T> friend struct A<T>::Inner; // expected-warning {{not supported}}
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
    friend void X<V>::operator+=(U); // expected-warning {{not supported}}
  };

  void test() {   
    X<int>() += 1.0;
  }
}

namespace test5 {
  template<template <class> class T> struct A {
    template<template <class> class U> friend void A<U>::foo(); // expected-warning {{not supported}}
  };

  template <class> struct B {};
  template class A<B>;
}
