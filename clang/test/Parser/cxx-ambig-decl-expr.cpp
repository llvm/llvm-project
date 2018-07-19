// RUN: %clang_cc1 -fsyntax-only -verify %s

struct X {
  template<typename T, typename U>
  static void f(int, int);
};

void f() {
  void (*ptr)(int, int) = &X::f<int, int>;

  unknown *p = 0; // expected-error {{unknown type name 'unknown'}}
  unknown * p + 0; // expected-error {{undeclared identifier 'unknown'}}
}
