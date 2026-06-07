// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  template <class T> class A {
    class Member {};
  };

  class B {
    template <class T> friend class A<T>::Member;
    int n;
  };

  A<int> a;
  B b;
}

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

namespace test2 {
  template <class T> struct A;

  class C {
    static void foo(); // #test2-C-foo
    template <class T> friend void A<T>::g();
  };

  template <class T> struct A {
    void f() { C::foo(); }
    // expected-error@-1 {{'foo' is a private member of 'test2::C'}}
    //   expected-note@#test2-C-foo {{implicitly declared private here}}
  };

  template <class T> struct A<T*> {
    void f() { C::foo(); }
    // expected-error@-1 {{'foo' is a private member of 'test2::C'}}
    //   expected-note@#test2-C-foo {{implicitly declared private here}}
  };

  template <> struct A<char> {
    void f() { C::foo(); }
    // expected-error@-1 {{'foo' is a private member of 'test2::C'}}
    //   expected-note@#test2-C-foo {{implicitly declared private here}}
  };
}

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
    template<template <class> class U> friend void A<U>::foo();
  };

  template <class> struct B {};
  template class A<B>;
}

namespace test6 {
  template <class T> struct A {
    struct B {
      static int f();
    };
  };

  struct C {
    int n;
    template <class T> friend struct A<T>::B;
  };

  template <class T> int A<T>::B::f() {
    C c;
    c.n = 0;
    return 0;
  }

  int k = A<int>::B::f();
}

namespace test7 {
  template <class T> struct A {
    struct D {
      void g();
    };
  };

  struct C {
    template <class T> friend void A<T>::D::g();
    // expected-error@-1 {{friend declaration does not name a member of a class template specialization}}
  };
}

namespace test8 {
  template <class T> struct A { // #test8-A
    T h();
  };

  template <> struct A<int> {
    int h();
  };

  template <> struct A<float *> {
    int *h();
  };

  class C {
    int n; // #test8-C-n
    template <class T> friend int *A<T *>::h();
  };

  template <class T> T A<T>::h() {
    return T();
  }

  int A<int>::h() {
    C c;
    c.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test8::C'}}
    //   expected-note@#test8-C-n {{implicitly declared private here}}
    //   expected-note@#test8-A {{candidate friend template ignored: could not match 'T *' against 'int'}}
    return 0;
  }

  template <> int *A<int *>::h() {
    C c;
    c.n = 0;
    return nullptr;
  }

  int *A<float *>::h() {
    C c;
    c.n = 0;
    return nullptr;
  }

  int *t1 = A<int *>().h();
  int *t2 = A<float *>().h();
  int t3 = A<int>().h();
}

namespace test9 {
  template <class T> struct A {
    template <T U> T i();
  };

  template <> struct A<int> {
    template <int U> int i();
  };

  struct C {
    int n;
    template <class T> template <T U> friend T A<T>::i();
  };

  template <class T> template <T U> T A<T>::i() {
    C c;
    c.n = 0;
    return U;
  }

  template <int U> int A<int>::i() {
    C c;
    c.n = 0;
    return U;
  }

  int x = A<int>().i<1>();
}

namespace test10 {
  template <class T> struct A;
  class C {
    static void foo(); // #test10-C-foo
    template <class T> friend void A<T>::f();
  };

  template <class T> struct A {
    void f() { C::foo(); }
  };

  template <> struct A<int> {
    int f() {
      C::foo();
      // expected-error@-1 {{'foo' is a private member of 'test10::C'}}
      //   expected-note@#test10-C-foo {{implicitly declared private here}}
      return 0;
    }
  };
}

namespace test11 {
  template <class> struct C;
  template <class T> struct A {
    template <class> struct B;
  };
  template <class T> struct D : A<T> {
    using A<T>::B;
  };

  template <class T> struct C {
    int n;
    template <class U> friend struct D<T>::B;
  };

  template <> template <class U> struct A<int>::B {
    static int f(C<int> &c) {
      c.n = 0;
      return 0;
    }
  };

  int x = A<int>::B<void>::f(*new C<int>);
}

namespace test12 {
  template <class T> struct A {
    template <T> struct B {
      static int f();
    };
  };

  template <class T> struct C {
    int n;
    template <class U> template <U V> friend struct A<U>::B;
  };

  template <class T> template <T V> int A<T>::B<V>::f() {
    C<T> c;
    c.n = 0;
    return 0;
  }

  int x = A<int>::B<0>::f();
}

namespace test13 {
  template <typename T> struct S {
    template <typename> friend class T::template X<int>::Y;
    // expected-error@-1 {{friend declaration does not name a member of a class template specialization}}
  };
}

namespace test14 {
  template <class T> struct A {
    template <bool V> struct B {
      static int f(B<false> &x) { return x.n; }

    private:
      int n;
      template <bool> friend struct A<T>::B;
    };
  };

  int x = A<int>::B<true>::f(*new A<int>::B<false>);
}

namespace test15 {
  template <class T> struct A {
    T f();
  };

  template <> struct A<int> {
    void f();
  };

  class C {
    int n; // #test15-C-n
    template <class T> friend T A<T>::f();
  };

  void A<int>::f() {
    C c;
    c.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test15::C'}}
    //   expected-note@#test15-C-n {{implicitly declared private here}}
  }
}

namespace test16 {
  template <class T> struct A {
    template <T U> T i();
  };

  template <> struct A<int> {
    template <int U> void i();
  };

  class C {
    int n; // #test16-C-n
    template <class T> template <T U> friend T A<T>::i();
  };

  template <int U> void A<int>::i() {
    C c;
    c.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test16::C'}}
    //   expected-note@#test16-C-n {{implicitly declared private here}}
  }
}
