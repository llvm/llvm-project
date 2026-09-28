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

namespace GH173728 {
#if __SIZEOF_SIZE_T__ == 8
int reduced() {
  int i;
  return ({
    struct T {
    } s[-sizeof(0)][0 == sizeof(i < 0)]; // expected-error {{array is too large}}
    0;
  });
}

int original() {
  int i = 0;
  return 1 + ({
    struct tree_el {
      int val;
      struct tree_el **right, *left;
    } state_t[1 + -(sizeof(0x1c))][0 == sizeof(sizeof(i))]; // expected-error {{array is too large}}
    0x97 < 10000;
  });
}

int too_large() {
  return 1 + ({ struct T {} s[(1ULL << 33) - 1][0]; 0x97 < 10000; });
}

signed char too_large_no_fold() {
  return ({ struct T {} s[(1ULL << 33) - 1][0]; 1000; });
}
#endif

int over_limit() {
  return 1 + ({ struct T {} s[0xFFFFFFFFu][0]; 0x97 < 10000; });
}

void small() {
  signed char a = ({ struct T {} s[4]; 1000; }); // expected-warning {{implicit conversion from 'int' to 'signed char' changes value from 1000 to -24}}
  signed char b = ({ struct T {} s[4][0]; 1000; }); // expected-warning {{implicit conversion from 'int' to 'signed char' changes value from 1000 to -24}}
}
}
