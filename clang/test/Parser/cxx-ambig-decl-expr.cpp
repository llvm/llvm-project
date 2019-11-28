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

// Disambiguating an array declarator from an array subscripting.
void arr() {
  int x[] = {1}; // expected-note 2{{previous}}

  // This is array indexing not an array declarator because a comma expression
  // is not syntactically a constant-expression.
  int(x[1,1]); // expected-warning 2{{unused}}

  // This is array indexing not an array declaration because a braced-init-list
  // is not syntactically a constant-expression.
  int(x[{0}]); // expected-error {{array subscript is not an integer}}
  struct A {
    struct Q { int n; };
    int operator[](Q);
  } a;
  int(a[{0}]); // expected-warning {{unused}}

  // These are array declarations.
  int(x[(1,1)]); // expected-error {{redefinition}}
  int(x[true ? 1,1 : 1]); // expected-error {{redefinition}}
}
