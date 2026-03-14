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
}
