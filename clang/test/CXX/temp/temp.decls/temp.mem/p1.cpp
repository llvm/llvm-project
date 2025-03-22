// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

template <class T> struct A {
  static T cond;

  template <class U> struct B {
    static T twice(U value) {
      return (cond ? value + value : value);
    }
  };
};
extern template bool A<bool>::cond;

int foo() {
  A<bool>::cond = true;
  return A<bool>::B<int>::twice(4);
}

namespace PR6376 {
  template<typename T>
  struct X {
    template<typename Y>
    struct Y1 { }; //
  };

  template<>
  struct X<float> {
    template<typename Y>
    struct Y1 { };
  };

  template<typename T, typename U>
  struct Z : public X<T>::template Y1<U> { };

  Z<float, int> z0;
}

namespace OutOfLine {
  template<typename T>
  struct A {
    struct B { };

    template<typename U, B V>
    void f();

    template<typename U, B V>
    void g() { } // expected-note {{previous definition is here}}

    template<typename U, B V>
    static int x;

    template<typename U, B V>
    static int x<U*, V>;

    template<typename U, B V>
    static inline int x<U&, V> = 0; // expected-note {{previous definition is here}}

    template<typename U, B V>
    struct C;

    template<typename U, B V>
    struct C<U*, V>;

    template<typename U, B V>
    struct C<U&, V> { }; // expected-note {{previous definition is here}}
  };

  template<typename T>
  template<typename U, typename A<T>::B V>
  void A<T>::f() { }

  template<typename T>
  template<typename U, typename A<T>::B V>
  void A<T>::g() { } // expected-error {{redefinition of 'g'}}

  template<typename T>
  template<typename U, typename A<T>::B V>
  int A<T>::x = 0;

  template<typename T>
  template<typename U, typename A<T>::B V>
  int A<T>::x<U*, V> = 0;

  template<typename T>
  template<typename U, typename A<T>::B V>
  int A<T>::x<U&, V> = 0; // expected-error {{redefinition of 'x<U &, V>'}}

  template<typename T>
  template<typename U, typename A<T>::B V>
  struct A<T>::C { };

  template<typename T>
  template<typename U, typename A<T>::B V>
  struct A<T>::C<U*, V> { };

  template<typename T>
  template<typename U, typename A<T>::B V>
  struct A<T>::C<U&, V> { }; // expected-error {{redefinition of 'C<U &, V>'}}

  // FIXME: Crashes when parsing the non-type template parameter prior to C++20
  template<>
  template<typename U, A<int>::B V>
  void A<int>::f() { }

  template<>
  template<typename U, A<int>::B V>
  void A<int>::g() { } // expected-note {{previous definition is here}}

  template<>
  template<typename U, A<int>::B V>
  void A<int>::g() { } // expected-error {{redefinition of 'g'}}

  template<>
  template<typename U, A<int>::B V>
  int A<int>::x = 0;

  template<>
  template<typename U, A<int>::B V>
  int A<int>::x<U*, V> = 0;

  template<>
  template<typename U, A<int>::B V>
  int A<int>::x<U&, V> = 0; // expected-note {{previous definition is here}}

  template<>
  template<typename U, A<int>::B V>
  int A<int>::x<U&, V> = 0; // expected-error {{redefinition of 'x<U &, V>'}}

  template<>
  template<typename U, A<int>::B V>
  struct A<int>::C { };

  template<>
  template<typename U, A<int>::B V>
  struct A<int>::C<U*, V> { };

  template<>
  template<typename U, A<int>::B V>
  struct A<int>::C<U&, V> { }; // expected-note {{previous definition is here}}

  template<>
  template<typename U, A<int>::B V>
  struct A<int>::C<U&, V> { }; // expected-error {{redefinition of 'C<U &, V>'}}


  template<class T>
  class X { // expected-note 2 {{X defined here}}
    using A = int;
    template<A> void a1();
    template<A> static int a2;
    template<A> class a3;
    template<OutOfLine::X<T>::A> void a4();
    template<OutOfLine::X<T>::A> static int a5;
    template<OutOfLine::X<T>::A> class a6;

    using B = int[];
    template<B> void b1();
    template<B> static int b2;
    template<B> class b3;
    template<OutOfLine::X<T>::B> void b4();
    template<OutOfLine::X<T>::B> static int b5;
    template<OutOfLine::X<T>::B> class b6;

    using Bad = int&&;
    template<Bad> void bad1();                      // expected-error {{non-type template parameter has rvalue reference type}}
    template<Bad> static int bad2;                  // expected-error {{non-type template parameter has rvalue reference type}}
    template<Bad> class bad3;                       // expected-error {{non-type template parameter has rvalue reference type}}
    template<OutOfLine::X<T>::Bad> void bad4();     // expected-error {{non-type template parameter has rvalue reference type}}
    template<OutOfLine::X<T>::Bad> static int bad5; // expected-error {{non-type template parameter has rvalue reference type}}
    template<OutOfLine::X<T>::Bad> class bad6;      // expected-error {{non-type template parameter has rvalue reference type}}

    template<const T> class Q;
    template<const T...> class Qp;

    template<int> void good();
  };

  template<class T> template<X<T>::A> void X<T>::a1() {}
  template<class T> template<X<T>::A> int X<T>::a2 = 2;
  template<class T> template<X<T>::A> class X<T>::a3 {};
  template<class T> template<X<T>::A> void X<T>::a4() {}
  template<class T> template<X<T>::A> int X<T>::a5 = 5;
  template<class T> template<X<T>::A> class X<T>::a6 {};

  template<class T> template<X<T>::B> void X<T>::b1() {}
  template<class T> template<X<T>::B> int X<T>::b2 = 2;
  template<class T> template<X<T>::B> class X<T>::b3 {};
  template<class T> template<X<T>::B> void X<T>::b4() {}
  template<class T> template<X<T>::B> int X<T>::b5 = 5;
  template<class T> template<X<T>::B> class X<T>::b6 {};

  template<class T> template<X<T>::Bad> void X<T>::bad1() {}
  template<class T> template<X<T>::Bad> int X<T>::bad2 = 2;
  template<class T> template<X<T>::Bad> class X<T>::bad3 {};
  template<class T> template<X<T>::Bad> void X<T>::bad4() {}
  template<class T> template<X<T>::Bad> int X<T>::bad5 = 5;
  template<class T> template<X<T>::Bad> class X<T>::bad6 {};

  template<class T> template<const T> class X<T>::Q {};
  template<class T> template<const T...> class X<T>::Qp {};

  template<class T>
  template<X<T>::Bad>
  void X<T>::good() {} // expected-error {{out-of-line definition of 'good' does not match any declaration in 'X<T>'}}

  template<class> using RRef = int&&;

  template<class T>
  template<RRef<T>> // expected-error {{non-type template parameter has rvalue reference type 'RRef<T>' (aka 'int &&')}}
  void X<T>::good() {} // expected-error {{out-of-line definition of 'good' does not match any declaration in 'X<T>'}}
}
