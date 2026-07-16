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
  template <class T> struct A {
    template <class U> void operator+=(U);

    template <class V>
    template <class U>
    friend void A<V>::operator+=(U);
  };

  void test() {
    A<int>() += 1.0;
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
    template <typename> friend class T::template A<int>::B;
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

  struct C {
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
  class D;

  template <class T> struct A {
    template <class, class U> struct B {
      struct C {
        static void f(D &);
      };
      static void g(D &);
    };
  };

  class D {
    int n;

    template <class T>
    template <class U>
    friend struct A<T>::B<T, U>::C;

    template <class T>
    template <class U>
    friend void A<T>::B<T, U>::g(D &);
  };

  template <class T>
  template <class V, class U>
  void A<T>::B<V, U>::C::f(D &x) {
    x.n = 0;
  }

  template <class T>
  template <class V, class U>
  void A<T>::B<V, U>::g(D &x) {
    x.n = 0;
  }

  template struct A<int>::B<int, double>;
}

namespace test21 {
  class C;

  template <class T> struct A {
    template <class U> struct B;
  };

  class C {
    int n; // #test21-C-n
    template <class T>
    template <class U>
    friend struct A<T>::B;
  };

  template <> struct A<int> {
    template <int U> struct B {
      static void f(C &x) {
        x.n = 0;
        // expected-error@-1 {{'n' is a private member of 'test21::C'}}
        //   expected-note@#test21-C-n {{implicitly declared private here}}
      }
    };
  };

  template struct A<int>::B<0>;
}

namespace test22 {
  class C;

  template <class T> struct A {
    struct B;
  };

  class C {
    int n; // #test22-C-n
    template <class T> friend struct A<T>::B;
  };

  template <> struct A<int> {
    union B {
      static void f(C &x) {
        x.n = 0;
        // expected-error@-1 {{'n' is a private member of 'test22::C'}}
        //   expected-note@#test22-C-n {{implicitly declared private here}}
      }
    };
  };
}

namespace test23 {
  class B;

  template <class T> struct A {
    template <class U> static void f(B &);
  };

  class B {
    int n; // #test23-B-n
    template <class T>
    template <class U>
    friend void A<T>::f(B &);
  };

  template <> struct A<int> {
    template <int U> static void f(B &x) {
      x.n = 0;
      // expected-error@-1 {{'n' is a private member of 'test23::B'}}
      //   expected-note@#test23-B-n {{implicitly declared private here}}
    }
  };

  template void A<int>::f<0>(B &);
}

namespace test24 {
  class B;

  template <class T> struct A {
    static void f(B &);
  };

  class B {
    int n; // #test24-B-n
    template <class T> friend void A<T>::f(B &);
  };

  template <> struct A<int> {
    static void f(B &x, ...) {
      x.n = 0;
      // expected-error@-1 {{'n' is a private member of 'test24::B'}}
      //   expected-note@#test24-B-n {{implicitly declared private here}}
    }
  };
}

namespace test25 {
  class C;

  template <class... Ts> struct A {
    struct B {
      static void f(C &);
    };
  };

  class C {
    int n;
    template <class... Ts> friend struct A<Ts...>::B;
  };

  template <class... Ts> void A<Ts...>::B::f(C &x) {
    x.n = 0;
  }

  template void A<int, double>::B::f(C &);
}

namespace test26 {
  class B;

  template <class T> struct A {
    template <class U> static void f(U);
  };

  template <> struct A<int> {
    template <class U> static void f(U);
  };

  class B {
    int n;
    template <class T>
    template <class U>
    friend void A<T>::f(U);
  };

  template <class T>
  template <class U>
  void A<T>::f(U) {
    B b;
    b.n = 0;
  }

  template <class U>
  void A<int>::f(U) {
    B b;
    b.n = 0;
  }

  template void A<long>::f<double>(double);
  template void A<int>::f<double>(double);
}

namespace test27 {
  template <class T> struct A {
    struct B {
      template <class U> static void g(U);
      template <class U> struct C;
    };
  };

  template <class V> struct D {
    template <class U> friend void A<V>::B::g(U);
    // expected-error@-1 {{nested name specifier 'A<V>::B' in friend declaration must end with a simple-template-id naming a class template, but 'B' is a non-template member}}

    template <class U> friend class A<V>::B::C;
    // expected-error@-1 {{nested name specifier 'A<V>::B' in friend declaration must end with a simple-template-id naming a class template, but 'B' is a non-template member}}
  };
}

namespace test28 {
  class B;

  template <class T> struct A {
    ~A();
  };

  template <> struct A<int> {
    ~A();
  };

  class B {
    int n;
    template <class T> friend A<T>::~A();
  };

  template <class T> A<T>::~A() {
    B b;
    b.n = 0;
  }

  A<int>::~A() {
    B b;
    b.n = 0;
  }

  template struct A<long>;
}

namespace test29 {
  template <class T> class D;

  template <class T> struct A {
    template <class I> struct B {
      template <class U> struct C {
        static void h(D<T> &);
      };
    };
  };

  template <class V> class D {
    int n;

    template <class U>
    friend class A<V>::B<int>::C;
  };

  template <class T>
  template <class I>
  template <class U>
  void A<T>::B<I>::C<U>::h(D<T> &x) {
    x.n = 0;
  }

  template struct A<int>::B<int>::C<double>;
}

namespace test30 {
  class B;

  template <class T> struct A {
    template <class U> static void f(U);
  };

  template <> struct A<int> {
    template <class U> static void f(U *);
  };

  class B {
    int n; // #test30-B-n
    template <class T>
    template <class U>
    friend void A<T>::f(U);
  };

  template <class U>
  void A<int>::f(U *) {
    B b;
    b.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test30::B'}}
    //   expected-note@#test30-B-n {{implicitly declared private here}}
  }

  template void A<int>::f<double>(double *);
}

namespace test31 {
  class B;

  template <class T> struct A {
    static void f(B &) noexcept;
  };

  template <> struct A<int> {
    static void f(B &);
  };

  class B {
    int n; // #test31-B-n
    template <class T> friend void A<T>::f(B &) noexcept;
  };

  template <class T> void A<T>::f(B &x) noexcept {
    x.n = 0;
  }

  void A<int>::f(B &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test31::B'}}
    //   expected-note@#test31-B-n {{implicitly declared private here}}
  }

  template void A<long>::f(B &) noexcept;
}

namespace test32 {
  class B;

  template <class T> struct A {
    A();
    operator int();
  };

  template <> struct A<char> {
    A();
    operator int();
  };

  class B {
    int n;
    template <class T> friend A<T>::A();
    template <class T> friend A<T>::operator int();
  };

  template <class T> A<T>::A() {
    B b;
    b.n = 0;
  }

  template <class T> A<T>::operator int() {
    B b;
    b.n = 0;
    return 0;
  }

  A<char>::A() {
    B b;
    b.n = 0;
  }

  A<char>::operator int() {
    B b;
    b.n = 0;
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

  template <class P> struct D {
    template <class U> friend struct P::template D<U>::M;
    // expected-error@-1 {{nested name specifier 'test33::C::template D<U>' in friend declaration must end with a simple-template-id naming a class template, but 'D' is an alias template}}
  };

  template struct D<A>;
  template struct D<C>;
  // expected-note@-1 {{in instantiation of template class 'test33::D<test33::C>' requested here}}

  template <class P> struct E {
    template <class U> friend struct P::template D<int>::N;
    // expected-error@-1 {{nested name specifier 'test33::C::template D<int>' in friend declaration must end with a simple-template-id naming a class template, but 'D' is an alias template}}
  };

  template struct E<A>;
  template struct E<C>;
  // expected-note@-1 {{in instantiation of template class 'test33::E<test33::C>' requested here}}

  template <class P> struct F {
    template <class U> friend void P::template D<U>::f();
    // expected-error@-1 {{nested name specifier 'test33::C::template D<U>' in friend declaration must end with a simple-template-id naming a class template, but 'D' is an alias template}}
  };

  template struct F<A>;
  template struct F<C>;
  // expected-note@-1 {{in instantiation of template class 'test33::F<test33::C>' requested here}}
}

namespace test34 {
  template <class V> class C;

  template <class T> struct A {
    template <class U> struct B {
      static void f(C<long> &);
    };
  };

  template <class V> class C {
    int n; // #test34-C-n

    template <class T>
    friend struct A<T>::template B<V>;
  };

  template class C<long>;

  template <class T>
  template <class U>
  void A<T>::B<U>::f(C<long> &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test34::C<long>'}}
    //   expected-note@#test34-C-n {{implicitly declared private here}}
  }

  template struct A<char>::B<long>;
  template struct A<char>::B<int>;
  // expected-note@-1 {{in instantiation of member function 'test34::A<char>::B<int>::f' requested here}}
}

namespace test35 {
  template <class V> class B;

  template <class T> struct A {
    template <class U, class V> static void f(B<V> &);
  };

  template <class V> class B {
    int n; // #test35-B-n

    template <class T>
    friend void A<T>::template f<int, V>(B &);
  };

  template <class T>
  template <class U, class V>
  void A<T>::f(B<V> &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test35::B<long>'}}
    //   expected-note@#test35-B-n {{implicitly declared private here}}
  }

  template class B<long>;
  template void A<char>::f<int, long>(B<long> &);
  template void A<char>::f<double, long>(B<long> &);
  // expected-note@-1 {{in instantiation of function template specialization 'test35::A<char>::f<double, long>' requested here}}
}

namespace test36 {
  template <class V> class B;

  template <class T> struct A {
    template <class U, class V> static void f(B<V> &, U);
  };

  template <class V> class B {
    int n; // #test36-B-n

    template <class T>
    friend void A<T>::f(B &, int);
  };

  template <class T>
  template <class U, class V>
  void A<T>::f(B<V> &x, U) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test36::B<long>'}}
    //   expected-note@#test36-B-n {{implicitly declared private here}}
  }

  template class B<long>;
  template void A<char>::f<int, long>(B<long> &, int);
  template void A<char>::f<double, long>(B<long> &, double);
  // expected-note@-1 {{in instantiation of function template specialization 'test36::A<char>::f<double, long>' requested here}}
}

namespace test37 {
  class C;

  template <class T> struct A {
    template <class U> struct B;
  };

  template <> struct A<int> {
    template <class U> struct B {
      static void f(C &);
    };
  };

  class C {
    int n; // #test37-C-n

    template <class T>
    friend struct A<T>::B<long>;
  };

  template <class U>
  void A<int>::B<U>::f(C &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test37::C'}}
    //   expected-note@#test37-C-n {{implicitly declared private here}}
  }

  template struct A<int>::B<long>;
  template struct A<int>::B<double>;
  // expected-note@-1 {{in instantiation of member function 'test37::A<int>::B<double>::f' requested here}}
}

namespace test38 {
  class B;

  template <class T> struct A {
    template <class U> static void f(B &);
  };

  template <> struct A<int> {
    template <class U> static void f(B &);
  };

  class B {
    int n; // #test38-B-n

    template <class T>
    friend void A<T>::f<long>(B &);
  };

  template <class U>
  void A<int>::f(B &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test38::B'}}
    //   expected-note@#test38-B-n {{implicitly declared private here}}
  }

  template void A<int>::f<long>(B &);
  template void A<int>::f<double>(B &);
  // expected-note@-1 {{in instantiation of function template specialization 'test38::A<int>::f<double>' requested here}}
}

namespace test39 {
  class D;

  template <class T> struct A {
    template <class U> struct B;
  };

  template <> struct A<int> {
    struct B {
      static void f(D &);
    };
  };

  template <> struct A<long> {
    template <class U> union B {
      static void f(D &);
    };
  };

  template <> struct A<char> {
    template <class U> struct C {
      static void f(D &);
    };
  };

  class D {
    int n; // #test39-D-n

    template <class T>
    friend struct A<T>::B<double>;
  };

  void A<int>::B::f(D &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test39::D'}}
    //   expected-note@#test39-D-n {{implicitly declared private here}}
  }

  template <class U>
  void A<long>::B<U>::f(D &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test39::D'}}
    //   expected-note@#test39-D-n {{implicitly declared private here}}
  }

  template <class U>
  void A<char>::C<U>::f(D &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test39::D'}}
    //   expected-note@#test39-D-n {{implicitly declared private here}}
  }
}

namespace test40 {
  template <class> class C;

  template <class T> struct A {
    template <class U> struct B;
  };

  template <> struct A<int> {
    struct B { // expected-note {{previous definition is here}}
      static void f(C<int> &);
    };
  };

  template <> struct A<long> {
    template <class U> struct B {
      static void f(C<long> &);
    };
  };

  template <class V> class C {
    int n; // #test40-C-n

    template <class U>
    friend struct A<V>::B; // expected-error {{redefinition of 'B' as different kind of symbol}}
  };

  template class C<int>;
  // expected-note@-1 {{in instantiation of template class 'test40::C<int>' requested here}}

  void A<int>::B::f(C<int> &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test40::C<int>'}}
    //   expected-note@#test40-C-n {{implicitly declared private here}}
  }

  template class C<long>;

  template <class U>
  void A<long>::B<U>::f(C<long> &x) {
    x.n = 0;
  }

  template struct A<long>::B<double>;
}

namespace test41 {
  class C;

  template <class T> struct A {
    struct B;
  };

  template <> struct A<int> {
    template <class U> struct B {
      static void f(C &);
    };
  };

  class C {
    int n; // #test41-C-n

    template <class T>
    friend struct A<T>::B;
  };

  template <class U>
  void A<int>::B<U>::f(C &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test41::C'}}
    //   expected-note@#test41-C-n {{implicitly declared private here}}
  }

  template struct A<int>::B<double>;
}

namespace test42 {
  template <class> class D;

  struct A {
    template <class U> struct B {
      struct C {
        static void f(D<A> &);
      };
      static void g(D<A> &);
    };
  };

  template <class T> class D {
    int n;

    template <class U>
    friend struct T::template B<U>::C;

    template <class U>
    friend void T::template B<U>::g(D &);
  };

  template <class U>
  void A::B<U>::C::f(D<A> &x) {
    x.n = 0;
  }

  template <class U>
  void A::B<U>::g(D<A> &x) {
    x.n = 0;
  }

  template struct A::B<int>;
}
