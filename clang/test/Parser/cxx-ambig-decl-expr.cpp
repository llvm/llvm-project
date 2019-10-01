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

auto (*p)() -> int(nullptr);
auto (*q)() -> int(*)(unknown); // expected-error {{unknown type name 'unknown'}}
auto (*r)() -> int(*)(unknown + 1); // expected-error {{undeclared identifier 'unknown'}}

int f(unknown const x); // expected-error {{unknown type name 'unknown'}}
