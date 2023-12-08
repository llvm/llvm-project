// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> T f0(T); // expected-note{{candidate function}}
int f0(int); // expected-note{{candidate function}}

void test_f0() {
  int (*fp0)(int) = f0;
  int (*fp1)(int) = &f0;
  float (*fp2)(float) = &f0;
}

namespace N {
  int f0(int); // expected-note{{candidate function}}
}

void test_f0_2() {
  using namespace N;
  int (*fp0)(int) = f0; // expected-error{{address of overloaded function 'f0' is ambiguous}}
  float (*fp1)(float) = f0;
}
