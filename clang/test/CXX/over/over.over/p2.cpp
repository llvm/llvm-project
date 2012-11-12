// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> T f0(T, T); //expected-note{{candidate}}

void test_f0() {
  int (*f0a)(int, int) = f0;
  int (*f0b)(int, int) = &f0;
  int (*f0c)(int, float) = f0; // expected-error{{address of overloaded function 'f0' does not match required type 'int (int, float)'}}
}
