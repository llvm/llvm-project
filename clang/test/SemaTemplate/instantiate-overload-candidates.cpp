// RUN: %clang_cc1 -fsyntax-only -verify %s

// This is the function actually selected during overload resolution, and the
// only one defined.
template <typename T> void f(T*, int) {}

template <typename T> struct S;
template <typename T> struct S_ : S<T> { typedef int type; }; // expected-note{{in instantiation}}
template <typename T> struct S {
  // Force T to have a complete type here so we can observe instantiations with
  // incomplete types.
  T t; // expected-error{{field has incomplete type}}
};

// Provide a bad class and an overload that instantiates templates with it.
class NoDefinition; // expected-note{{forward declaration}}
template <typename T> S_<NoDefinition>::type f(T*, NoDefinition*); // expected-note{{in instantiation}}

void test(int x) {
  f(&x, 0);
}

// Ensure that we instantiate an overloaded function if it's selected by
// overload resolution when initializing a function pointer.
template<typename T> struct X {
  static T f() { T::error; } // expected-error {{has no members}}
  static T f(bool);
};
void (*p)() = &X<void>::f; // expected-note {{instantiation of}}

namespace PR13098 {
  struct A {
    A(int);
    void operator++() {}
    void operator+(int) {}
    void operator+(A) {}
    void operator[](int) {}
    void operator[](A) {}
  };
  struct B : A {
    using A::operator++;
    using A::operator+;
    using A::operator[];
  };
  template<typename T> void f(B b) {
    ++b;
    b + 0;
    b[0];
  }
  template void f<void>(B);
}
