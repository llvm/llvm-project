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
    // expected-error@-1 {{nested name specifier 'A<T>::D' in friend declaration must end with a simple-template-id naming a class template, but 'D' is a non-template member}}
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

namespace test17 {
  template <class T> class A;

  template <class T>
  struct B {
    template <bool V>
    struct C {
      int f(A<T> *p) { return p->x; }
    };
  };

  template <class T>
  class A {
    friend struct B<T>::template C<true>;
    int x;
  };

  template struct B<int>::C<true>;
}

namespace test18 {
  template <class T> class A;

  template <class T>
  struct B {
    template <bool V>
    struct C {
      int f(A<T> *p) { return p->x; }
      // expected-error@-1 {{'x' is a private member of 'test18::A<int>'}}
    };
  };

  template <class T>
  class A {
    friend struct B<T>::template C<false>;
    int x;
    // expected-note@-1 {{implicitly declared private here}}
  };

  template struct B<int>::C<true>;
  // expected-note@-1 {{in instantiation of member function 'test18::B<int>::C<true>::f' requested here}}
}

namespace test19 {
  template <class...> struct A {
    struct B;
    static void f();
  };

  struct X {
    template <class T, class U> // #test19-U-type
    friend struct A<T>::B;
    // expected-error@-1 {{template parameter of friend declaration cannot be deduced from 'A<T>'}}
    //   expected-note@#test19-U-type {{non-deducible template parameter 'U'}}

    template <class T, class U> // #test19-U-function
    friend void A<T>::f();
    // expected-error@-1 {{template parameter of friend declaration cannot be deduced from 'A<T>'}}
    //   expected-note@#test19-U-function {{non-deducible template parameter 'U'}}

    template <class... Ts> // #test19-Ts
    friend struct A<Ts..., int>::B;
    // expected-error@-1 {{template parameter of friend declaration cannot be deduced from 'A<Ts..., int>'}}
    //   expected-note@#test19-Ts {{non-deducible template parameter 'Ts'}}
  };
}

namespace test20 {
  class X;

  template <class T> struct A {
    template <class, class U> struct B {
      struct C {
        static void f(X &);
      };
      static void g(X &);
    };
  };

  class X {
    int n;

    template <class T>
    template <class U>
    friend struct A<T>::B<T, U>::C;

    template <class T>
    template <class U>
    friend void A<T>::B<T, U>::g(X &);
  };

  template <class T>
  template <class V, class U>
  void A<T>::B<V, U>::C::f(X &x) {
    x.n = 0;
  }

  template <class T>
  template <class V, class U>
  void A<T>::B<V, U>::g(X &x) {
    x.n = 0;
  }

  template struct A<int>::B<int, double>;
}

namespace test21 {
  class X;

  template <class T> struct A {
    template <class U> struct B;
  };

  class X {
    int n; // #test21-X-n
    template <class T>
    template <class U>
    friend struct A<T>::B;
  };

  template <> struct A<int> {
    template <int U> struct B {
      static void f(X &x) {
        x.n = 0;
        // expected-error@-1 {{'n' is a private member of 'test21::X'}}
        //   expected-note@#test21-X-n {{implicitly declared private here}}
      }
    };
  };

  template struct A<int>::B<0>;
}

namespace test22 {
  class X;

  template <class T> struct A {
    struct B;
  };

  class X {
    int n; // #test22-X-n
    template <class T> friend struct A<T>::B;
  };

  template <> struct A<int> {
    union B {
      static void f(X &x) {
        x.n = 0;
        // expected-error@-1 {{'n' is a private member of 'test22::X'}}
        //   expected-note@#test22-X-n {{implicitly declared private here}}
      }
    };
  };
}

namespace test23 {
  class X;

  template <class T> struct A {
    template <class U> static void f(X &);
  };

  class X {
    int n; // #test23-X-n
    template <class T>
    template <class U>
    friend void A<T>::f(X &);
  };

  template <> struct A<int> {
    template <int U> static void f(X &x) {
      x.n = 0;
      // expected-error@-1 {{'n' is a private member of 'test23::X'}}
      //   expected-note@#test23-X-n {{implicitly declared private here}}
    }
  };

  template void A<int>::f<0>(X &);
}

namespace test24 {
  class X;

  template <class T> struct A {
    static void f(X &);
  };

  class X {
    int n; // #test24-X-n
    template <class T> friend void A<T>::f(X &);
  };

  template <> struct A<int> {
    static void f(X &x, ...) {
      x.n = 0;
      // expected-error@-1 {{'n' is a private member of 'test24::X'}}
      //   expected-note@#test24-X-n {{implicitly declared private here}}
    }
  };
}

namespace test25 {
  class X;

  template <class... Ts> struct A {
    struct B {
      static void f(X &);
    };
  };

  class X {
    int n;
    template <class... Ts> friend struct A<Ts...>::B;
  };

  template <class... Ts> void A<Ts...>::B::f(X &x) {
    x.n = 0;
  }

  template void A<int, double>::B::f(X &);
}
