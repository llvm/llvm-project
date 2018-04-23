// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr727 { // dr727: partial
  struct A {
    template<typename T> struct C; // expected-note 6{{here}}
    template<typename T> void f(); // expected-note {{here}}
    template<typename T> static int N; // expected-error 0-1{{C++14}} expected-note 6{{here}}

    template<> struct C<int>;
    template<> void f<int>();
    template<> static int N<int>;

    template<typename T> struct C<T*>;
    template<typename T> static int N<T*>;

    struct B {
      template<> struct C<float>; // expected-error {{not in class 'A' or an enclosing namespace}}
      template<> void f<float>(); // expected-error {{no function template matches}}
      template<> static int N<float>; // expected-error {{not in class 'A' or an enclosing namespace}}

      template<typename T> struct C<T**>; // expected-error {{not in class 'A' or an enclosing namespace}}
      template<typename T> static int N<T**>; // expected-error {{not in class 'A' or an enclosing namespace}}

      template<> struct A::C<double>; // expected-error {{not in class 'A' or an enclosing namespace}}
      template<> void A::f<double>(); // expected-error {{no function template matches}} expected-error {{cannot have a qualified name}}
      template<> static int A::N<double>; // expected-error {{not in class 'A' or an enclosing namespace}} expected-error {{cannot have a qualified name}}

      template<typename T> struct A::C<T***>; // expected-error {{not in class 'A' or an enclosing namespace}}
      template<typename T> static int A::N<T***>; // expected-error {{not in class 'A' or an enclosing namespace}} expected-error {{cannot have a qualified name}}
    };
  };

  template<> struct A::C<char>;
  template<> void A::f<char>();
  template<> int A::N<char>;

  template<typename T> struct A::C<T****>;
  template<typename T> int A::N<T****>;

  namespace C {
    template<> struct A::C<long>; // expected-error {{not in class 'A' or an enclosing namespace}}
    template<> void A::f<long>(); // expected-error {{not in class 'A' or an enclosing namespace}}
    template<> int A::N<long>; // expected-error {{not in class 'A' or an enclosing namespace}}

    template<typename T> struct A::C<T*****>; // expected-error {{not in class 'A' or an enclosing namespace}}
    template<typename T> int A::N<T*****>; // expected-error {{not in class 'A' or an enclosing namespace}}
  }

  template<typename>
  struct D {
    template<typename T> struct C { typename T::error e; }; // expected-error {{no members}}
    template<typename T> void f() { T::error; } // expected-error {{no members}}
    template<typename T> static const int N = T::error; // expected-error 2{{no members}} expected-error 0-1{{C++14}}

    template<> struct C<int> {};
    template<> void f<int>() {}
    template<> static const int N<int>;

    template<typename T> struct C<T*> {};
    template<typename T> static const int N<T*>;
  };

  void d(D<int> di) {
    D<int>::C<int>();
    di.f<int>();
    int a = D<int>::N<int>; // FIXME: expected-note {{instantiation of}}

    D<int>::C<int*>();
    int b = D<int>::N<int*>;

    D<int>::C<float>(); // expected-note {{instantiation of}}
    di.f<float>(); // expected-note {{instantiation of}}
    int c = D<int>::N<float>; // expected-note {{instantiation of}}
  }
}

namespace dr777 { // dr777: 3.7
#if __cplusplus >= 201103L
template <typename... T>
void f(int i = 0, T ...args) {}
void ff() { f(); }

template <typename... T>
void g(int i = 0, T ...args, T ...args2) {}

template <typename... T>
void h(int i = 0, T ...args, int j = 1) {}
#endif
}
