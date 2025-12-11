// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

class Foo {
  ~Foo();
  Foo(const Foo&);
public:
  Foo(int);
};

class Bar {
  int foo_count;
  Foo foos[0];
#if __cplusplus >= 201103L
// expected-note@-2 {{copy constructor of 'Bar' is implicitly deleted because field 'foos' has an inaccessible copy constructor}}
#endif
  Foo foos2[0][2];
  Foo foos3[2][0];

public:
  Bar(): foo_count(0) { }
  ~Bar() { }
};

void testBar() {
  Bar b;
  Bar b2(b);
#if __cplusplus >= 201103L
// expected-error@-2 {{call to implicitly-deleted copy constructor of 'Bar}}
#else
// expected-no-diagnostics
#endif
  b = b2;
}

namespace GH170040 {
#if __cplusplus >= 202002L
template <int N> struct Foo {
    operator int() const requires(N == 2);
    template <int I = 0, char (*)[(I)] = nullptr> operator long() const;
};

void test () {
    Foo<2> foo;
    long bar = foo;
}
#endif
}
