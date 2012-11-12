// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR4364
template<class T> struct a {
  T b() {
    return typename T::x();
  }
};
struct B {
  typedef B x;
};
B c() {
  a<B> x;
  return x.b();
}

// Some extra tests for invalid cases
template<class T> struct test2 { T b() { return typename T::a; } }; // expected-error{{expected '(' for function-style cast or type construction}}
template<class T> struct test3 { T b() { return typename a; } }; // expected-error{{expected a qualified name after 'typename'}}
