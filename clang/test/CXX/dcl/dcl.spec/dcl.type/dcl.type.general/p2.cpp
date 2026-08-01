// RUN:  %clang_cc1 -std=c++2c -verify %s

void func1() {
  typedef float foo; // expected-note {{previous definition is here}}
  auto foo{16};      // expected-error {{redefinition of 'foo' as different kind of symbol}}
}

typedef float bar;
void func2() {
  auto bar{16};
}
