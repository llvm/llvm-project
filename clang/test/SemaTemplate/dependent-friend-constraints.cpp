// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

namespace test0 {
  class C;

  template <class T> struct A { // #test0-A
    struct B {
      static void f(C &);
    };
  };

  class C {
    int n; // #test0-C-n
    template <class T>
      requires __is_same(T, int) // #test0-requires
    friend struct A<T>::B;
  };

  template <class T> void A<T>::B::f(C &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test0::C'}}
    //   expected-note@#test0-C-n {{implicitly declared private here}}
    //   expected-note@#test0-A {{candidate template ignored: constraints not satisfied [with T = double]}}
    //   expected-note@#test0-requires {{because '__is_same(double, int)' evaluated to false}}
  }

  template void A<int>::B::f(C &);
  template void A<double>::B::f(C &);
  // expected-note@-1 {{in instantiation of member function 'test0::A<double>::B::f' requested here}}
}

namespace test1 {
  class C;

  template <class T> struct A {
    template <class U>
      requires __is_same(T, U)
    struct B;
  };

  class C {
    int n;
    template <class T>
    template <class U>
      requires __is_same(T, U)
    friend struct A<T>::B;
  };

  template <> struct A<int> {
    template <class U>
      requires __is_same(int, U)
    struct B {
      static void f(C &x) { x.n = 0; }
    };
  };

  template struct A<int>::B<int>;
}

namespace test2 {
  class C;

  template <class T> struct A {
    template <class U>
      requires __is_same(T, U)
    struct B;
  };

  class C {
    int n; // #test2-C-n
    template <class T>
    template <class U>
      requires __is_same(T, U)
    friend struct A<T>::B;
  };

  template <> struct A<int> {
    template <class U>
      requires (sizeof(U) != 0)
    struct B {
      static void f(C &x) {
        x.n = 0;
        // expected-error@-1 {{'n' is a private member of 'test2::C'}}
        //   expected-note@#test2-C-n {{implicitly declared private here}}
      }
    };
  };

  template struct A<int>::B<int>;
}

namespace test3 {
  class B;

  template <class T> struct A {
    template <class U>
      requires __is_same(T, U)
    static void f(B &);
  };

  class B {
    int n;
    template <class T>
    template <class U>
      requires __is_same(T, U)
    friend void A<T>::f(B &);
  };

  template <> struct A<int> {
    template <class U>
      requires __is_same(int, U)
    static void f(B &x) {
      x.n = 0;
    }
  };

  template void A<int>::f<int>(B &);
}

namespace test4 {
  class B;

  template <class T> struct A {
    template <class U>
      requires __is_same(T, U)
    static void f(B &);
  };

  class B {
    int n; // #test4-B-n
    template <class T>
    template <class U>
      requires __is_same(T, U)
    friend void A<T>::f(B &);
  };

  template <> struct A<int> {
    template <class U>
      requires (sizeof(U) != 0)
    static void f(B &x) {
      x.n = 0;
      // expected-error@-1 {{'n' is a private member of 'test4::B'}}
      //   expected-note@#test4-B-n {{implicitly declared private here}}
    }
  };

  template void A<int>::f<int>(B &);
}

namespace test5 {
  class B;

  template <class T> struct A {
    template <class U>
    static void f(B &) requires (sizeof(U) != 0);
  };

  class B {
    int n;
    template <class T>
    template <class U>
    friend void A<T>::f(B &) requires (sizeof(U) != 0);
  };

  template <> struct A<int> {
    template <class U>
    static void f(B &x) requires (sizeof(U) != 0) {
      x.n = 0;
    }
  };

  template void A<int>::f<int>(B &);
}

namespace test6 {
  class B;

  template <class T> struct A {
    template <class U>
    static void f(B &) requires (sizeof(U) != 0);
  };

  class B {
    int n; // #test6-B-n
    template <class T>
    template <class U>
    friend void A<T>::f(B &) requires (sizeof(U) != 0);
  };

  template <> struct A<int> {
    template <class U>
    static void f(B &x) requires (sizeof(U) > 1) {
      x.n = 0;
      // expected-error@-1 {{'n' is a private member of 'test6::B'}}
      //   expected-note@#test6-B-n {{implicitly declared private here}}
    }
  };

  template void A<int>::f<int>(B &);
}

namespace test7 {
  template <class T> struct A; // #test7-A

  template <class V> class C {
    int n; // #test7-C-n
    template <class T>
      requires __is_same(T, V) // #test7-requires
    friend struct A<T>::B;
  };

  template <> struct A<int> {
    struct B {
      static void f(C<int> &x) { x.n = 0; }
    };
  };

  template <> struct A<double> {
    struct B {
      static void f(C<int> &x) {
        x.n = 0;
        // expected-error@-1 {{'n' is a private member of 'test7::C<int>'}}
        //   expected-note@#test7-C-n {{implicitly declared private here}}
        //   expected-note@#test7-A {{candidate template ignored: constraints not satisfied [with T = double]}}
        //   expected-note@#test7-requires {{because '__is_same(double, int)' evaluated to false}}
      }
    };
  };
}

namespace test8 {
  template <class T, class U>
  concept Same = __is_same(T, U);

  class B;

  template <class T> struct A {
    template <class U>
    static void f(B &) requires Same<T, T>;
  };

  template <> struct A<int> {
    template <class U>
    static void f(B &) requires Same<U, U>;
  };

  class B {
    int n; // #test8-B-n

    template <class T>
    template <class U>
    friend void A<T>::f(B &) requires Same<T, T>;
  };

  template <class U>
  void A<int>::f(B &x) requires Same<U, U> {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test8::B'}}
    //   expected-note@#test8-B-n {{implicitly declared private here}}
  }

  template void A<int>::f<double>(B &);
}
