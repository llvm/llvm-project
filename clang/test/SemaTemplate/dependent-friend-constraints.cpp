// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

namespace test0 {
  class X;

  template <class T> struct A { // #test0-A
    struct B {
      static void f(X &);
    };
  };

  class X {
    int n; // #test0-X-n
    template <class T>
      requires __is_same(T, int) // #test0-requires
    friend struct A<T>::B;
  };

  template <class T> void A<T>::B::f(X &x) {
    x.n = 0;
    // expected-error@-1 {{'n' is a private member of 'test0::X'}}
    //   expected-note@#test0-X-n {{implicitly declared private here}}
    //   expected-note@#test0-A {{candidate template ignored: constraints not satisfied [with T = double]}}
    //   expected-note@#test0-requires {{because '__is_same(double, int)' evaluated to false}}
  }

  template void A<int>::B::f(X &);
  template void A<double>::B::f(X &);
  // expected-note@-1 {{in instantiation of member function 'test0::A<double>::B::f' requested here}}
}

namespace test1 {
  class X;

  template <class T> struct A {
    template <class U>
      requires __is_same(T, U)
    struct B;
  };

  class X {
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
      static void f(X &x) { x.n = 0; }
    };
  };

  template struct A<int>::B<int>;
}

namespace test2 {
  class X;

  template <class T> struct A {
    template <class U>
      requires __is_same(T, U)
    struct B;
  };

  class X {
    int n; // #test2-X-n
    template <class T>
    template <class U>
      requires __is_same(T, U)
    friend struct A<T>::B;
  };

  template <> struct A<int> {
    template <class U>
      requires (sizeof(U) != 0)
    struct B {
      static void f(X &x) {
        x.n = 0;
        // expected-error@-1 {{'n' is a private member of 'test2::X'}}
        //   expected-note@#test2-X-n {{implicitly declared private here}}
      }
    };
  };

  template struct A<int>::B<int>;
}

namespace test3 {
  class X;

  template <class T> struct A {
    template <class U>
      requires __is_same(T, U)
    static void f(X &);
  };

  class X {
    int n;
    template <class T>
    template <class U>
      requires __is_same(T, U)
    friend void A<T>::f(X &);
  };

  template <> struct A<int> {
    template <class U>
      requires __is_same(int, U)
    static void f(X &x) {
      x.n = 0;
    }
  };

  template void A<int>::f<int>(X &);
}

namespace test4 {
  class X;

  template <class T> struct A {
    template <class U>
      requires __is_same(T, U)
    static void f(X &);
  };

  class X {
    int n; // #test4-X-n
    template <class T>
    template <class U>
      requires __is_same(T, U)
    friend void A<T>::f(X &);
  };

  template <> struct A<int> {
    template <class U>
      requires (sizeof(U) != 0)
    static void f(X &x) {
      x.n = 0;
      // expected-error@-1 {{'n' is a private member of 'test4::X'}}
      //   expected-note@#test4-X-n {{implicitly declared private here}}
    }
  };

  template void A<int>::f<int>(X &);
}

namespace test5 {
  class X;

  template <class T> struct A {
    template <class U>
    static void f(X &) requires (sizeof(U) != 0);
  };

  class X {
    int n;
    template <class T>
    template <class U>
    friend void A<T>::f(X &) requires (sizeof(U) != 0);
  };

  template <> struct A<int> {
    template <class U>
    static void f(X &x) requires (sizeof(U) != 0) {
      x.n = 0;
    }
  };

  template void A<int>::f<int>(X &);
}

namespace test6 {
  class X;

  template <class T> struct A {
    template <class U>
    static void f(X &) requires (sizeof(U) != 0);
  };

  class X {
    int n; // #test6-X-n
    template <class T>
    template <class U>
    friend void A<T>::f(X &) requires (sizeof(U) != 0);
  };

  template <> struct A<int> {
    template <class U>
    static void f(X &x) requires (sizeof(U) > 1) {
      x.n = 0;
      // expected-error@-1 {{'n' is a private member of 'test6::X'}}
      //   expected-note@#test6-X-n {{implicitly declared private here}}
    }
  };

  template void A<int>::f<int>(X &);
}

namespace test7 {
  template <class T> struct A; // #test7-A

  template <class V> class X {
    int n; // #test7-X-n
    template <class T>
      requires __is_same(T, V) // #test7-requires
    friend struct A<T>::B;
  };

  template <> struct A<int> {
    struct B {
      static void f(X<int> &x) { x.n = 0; }
    };
  };

  template <> struct A<double> {
    struct B {
      static void f(X<int> &x) {
        x.n = 0;
        // expected-error@-1 {{'n' is a private member of 'test7::X<int>'}}
        //   expected-note@#test7-X-n {{implicitly declared private here}}
        //   expected-note@#test7-A {{candidate template ignored: constraints not satisfied [with T = double]}}
        //   expected-note@#test7-requires {{because '__is_same(double, int)' evaluated to false}}
      }
    };
  };
}
