// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++14 -fcxx-exceptions -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s

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

namespace test26 {
  class X;

  template <class T> struct A {
    template <class U> static void f(U);
  };

  template <> struct A<int> {
    template <class U> static void f(U);
  };

  class X {
    int n;
    template <class T>
    template <class U>
    friend void A<T>::f(U);
  };

  template <class T>
  template <class U>
  void A<T>::f(U) {
    X x;
    x.n = 0;
  }

  template <class U>
  void A<int>::f(U) {
    X x;
    x.n = 0;
  }

  template void A<long>::f<double>(double);
  template void A<int>::f<double>(double);
}

namespace test27 {
  template <class T> struct A {
    struct D {
      template <class U> static void g(U);
      template <class U> struct Y;
    };
  };

  template <class V> struct X {
    template <class U> friend void A<V>::D::g(U);
    // expected-error@-1 {{nested name specifier 'A<V>::D' in friend declaration must end with a simple-template-id naming a class template, but 'D' is a non-template member}}

    template <class U> friend class A<V>::D::Y;
    // expected-error@-1 {{nested name specifier 'A<V>::D' in friend declaration must end with a simple-template-id naming a class template, but 'D' is a non-template member}}
  };
}

namespace test28 {
  class X;

  template <class T> struct A {
    ~A();
  };

  template <> struct A<int> {
    ~A();
  };

  class X {
    int n;
    template <class T> friend A<T>::~A();
  };

  template <class T> A<T>::~A() {
    X x;
    x.n = 0;
  }

  A<int>::~A() {
    X x;
    x.n = 0;
  }

  template struct A<long>;
}

namespace test29 {
  template <class T> class X;

  template <class T> struct A {
    template <class I> struct D {
      template <class U> struct Y {
        static void h(X<T> &);
      };
    };
  };

  template <class V> class X {
    int n;

    template <class U>
    friend class A<V>::D<int>::Y;
  };

  template <class T>
  template <class I>
  template <class U>
  void A<T>::D<I>::Y<U>::h(X<T> &x) {
    x.n = 0;
  }

  template struct A<int>::D<int>::Y<double>;
}

namespace test30 {
  class X;

  template <class T> struct A {
    template <class U> static void f(U);
  };

  template <> struct A<int> {
    template <class U> static void f(U *);
  };

  class X {
    int n; // #test30-X-n
    template <class T>
    template <class U>
    friend void A<T>::f(U);
  };

  template <class U>
  void A<int>::f(U *) {
    X x;
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test30::X'}}
    //   expected-note@#test30-X-n {{implicitly declared private here}}
  }

  template void A<int>::f<double>(double *);
}

namespace test31 {
  class X;

  template <class T> struct A {
    static void f(X &) noexcept;
  };

  template <> struct A<int> {
    static void f(X &);
  };

  class X {
    int n; // #test31-X-n
    template <class T> friend void A<T>::f(X &) noexcept;
  };

  template <class T> void A<T>::f(X &x) noexcept {
    x.n = 0;
  }

  void A<int>::f(X &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test31::X'}}
    //   expected-note@#test31-X-n {{implicitly declared private here}}
  }

  template void A<long>::f(X &) noexcept;
}

namespace test32 {
  class X;

  template <class T> struct A {
    A();
    operator int();
  };

  template <> struct A<char> {
    A();
    operator int();
  };

  class X {
    int n;
    template <class T> friend A<T>::A();
    template <class T> friend A<T>::operator int();
  };

  template <class T> A<T>::A() {
    X x;
    x.n = 0;
  }

  template <class T> A<T>::operator int() {
    X x;
    x.n = 0;
    return 0;
  }

  A<char>::A() {
    X x;
    x.n = 0;
  }

  A<char>::operator int() {
    X x;
    x.n = 0;
    return 0;
  }

  template struct A<long>;
}

namespace test33 {
  struct A {
    template <class> struct D {
      struct M;
      template <class> struct N;
      static void f();
    };
  };

  struct B {
    struct M;
    template <class> struct N;
    static void f();
  };

  struct C {
    template <class> using D = B;
  };

  template <class P> struct X {
    template <class U> friend struct P::template D<U>::M;
    // expected-error@-1 {{nested name specifier 'test33::C::template D<U>' in friend declaration must end with a simple-template-id naming a class template, but 'D' is an alias template}}
  };

  template struct X<A>;
  template struct X<C>;
  // expected-note@-1 {{in instantiation of template class 'test33::X<test33::C>' requested here}}

  template <class P> struct Y {
    template <class U> friend struct P::template D<int>::N;
    // expected-error@-1 {{nested name specifier 'test33::C::template D<int>' in friend declaration must end with a simple-template-id naming a class template, but 'D' is an alias template}}
  };

  template struct Y<A>;
  template struct Y<C>;
  // expected-note@-1 {{in instantiation of template class 'test33::Y<test33::C>' requested here}}

  template <class P> struct Z {
    template <class U> friend void P::template D<U>::f();
    // expected-error@-1 {{nested name specifier 'test33::C::template D<U>' in friend declaration must end with a simple-template-id naming a class template, but 'D' is an alias template}}
  };

  template struct Z<A>;
  template struct Z<C>;
  // expected-note@-1 {{in instantiation of template class 'test33::Z<test33::C>' requested here}}
}

namespace test34 {
  template <class V> class X;

  template <class T> struct A {
    template <class U> struct B {
      static void f(X<long> &);
    };
  };

  template <class V> class X {
    int n; // #test34-X-n

    template <class T>
    friend struct A<T>::template B<V>;
  };

  template class X<long>;

  template <class T>
  template <class U>
  void A<T>::B<U>::f(X<long> &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test34::X<long>'}}
    //   expected-note@#test34-X-n {{implicitly declared private here}}
  }

  template struct A<char>::B<long>;
  template struct A<char>::B<int>;
  // expected-note@-1 {{in instantiation of member function 'test34::A<char>::B<int>::f' requested here}}
}

namespace test35 {
  template <class V> class X;

  template <class T> struct A {
    template <class U, class V> static void f(X<V> &);
  };

  template <class V> class X {
    int n; // #test35-X-n

    template <class T>
    friend void A<T>::template f<int, V>(X &);
  };

  template <class T>
  template <class U, class V>
  void A<T>::f(X<V> &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test35::X<long>'}}
    //   expected-note@#test35-X-n {{implicitly declared private here}}
  }

  template class X<long>;
  template void A<char>::f<int, long>(X<long> &);
  template void A<char>::f<double, long>(X<long> &);
  // expected-note@-1 {{in instantiation of function template specialization 'test35::A<char>::f<double, long>' requested here}}
}

namespace test36 {
  template <class V> class X;

  template <class T> struct A {
    template <class U, class V> static void f(X<V> &, U);
  };

  template <class V> class X {
    int n; // #test36-X-n

    template <class T>
    friend void A<T>::f(X &, int);
  };

  template <class T>
  template <class U, class V>
  void A<T>::f(X<V> &x, U) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test36::X<long>'}}
    //   expected-note@#test36-X-n {{implicitly declared private here}}
  }

  template class X<long>;
  template void A<char>::f<int, long>(X<long> &, int);
  template void A<char>::f<double, long>(X<long> &, double);
  // expected-note@-1 {{in instantiation of function template specialization 'test36::A<char>::f<double, long>' requested here}}
}

namespace test37 {
  class X;

  template <class T> struct A {
    template <class U> struct B;
  };

  template <> struct A<int> {
    template <class U> struct B {
      static void f(X &);
    };
  };

  class X {
    int n; // #test37-X-n

    template <class T>
    friend struct A<T>::B<long>;
  };

  template <class U>
  void A<int>::B<U>::f(X &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test37::X'}}
    //   expected-note@#test37-X-n {{implicitly declared private here}}
  }

  template struct A<int>::B<long>;
  template struct A<int>::B<double>;
  // expected-note@-1 {{in instantiation of member function 'test37::A<int>::B<double>::f' requested here}}
}

namespace test38 {
  class X;

  template <class T> struct A {
    template <class U> static void f(X &);
  };

  template <> struct A<int> {
    template <class U> static void f(X &);
  };

  class X {
    int n; // #test38-X-n

    template <class T>
    friend void A<T>::f<long>(X &);
  };

  template <class U>
  void A<int>::f(X &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test38::X'}}
    //   expected-note@#test38-X-n {{implicitly declared private here}}
  }

  template void A<int>::f<long>(X &);
  template void A<int>::f<double>(X &);
  // expected-note@-1 {{in instantiation of function template specialization 'test38::A<int>::f<double>' requested here}}
}

namespace test39 {
  class X;

  template <class T> struct A {
    template <class U> struct B;
  };

  template <> struct A<int> {
    struct B {
      static void f(X &);
    };
  };

  template <> struct A<long> {
    template <class U> union B {
      static void f(X &);
    };
  };

  template <> struct A<char> {
    template <class U> struct C {
      static void f(X &);
    };
  };

  class X {
    int n; // #test39-X-n

    template <class T>
    friend struct A<T>::B<double>;
  };

  void A<int>::B::f(X &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test39::X'}}
    //   expected-note@#test39-X-n {{implicitly declared private here}}
  }

  template <class U>
  void A<long>::B<U>::f(X &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test39::X'}}
    //   expected-note@#test39-X-n {{implicitly declared private here}}
  }

  template <class U>
  void A<char>::C<U>::f(X &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test39::X'}}
    //   expected-note@#test39-X-n {{implicitly declared private here}}
  }
}
