// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-bitwise-compare %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall -Wno-unused %s

void test(int x) {
  bool b1 = (8 & x) == 3;
  // expected-warning@-1 {{bitwise comparison always evaluates to false}}
  bool b2 = x | 5;
  // expected-warning@-1 {{bitwise or with non-zero value always evaluates to true}}
  bool b3 = (x | 5);
  // expected-warning@-1 {{bitwise or with non-zero value always evaluates to true}}
  bool b4 = !!(x | 5);
  // expected-warning@-1 {{bitwise or with non-zero value always evaluates to true}}
}

template <int I, class T>  // silence
void foo(int x) {
    bool b1 = (x & sizeof(T)) == 8;
    bool b2 = (x & I) == 8;
    bool b3 = (x & 4) == 8; // expected-warning {{bitwise comparison always evaluates to false}}
}

void run(int x) {
    foo<4, int>(8); // expected-note {{in instantiation of function template specialization 'foo<4, int>' requested here}}
}
