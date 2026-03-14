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
  int(x[1,1]); // expected-warning {{left operand of comma operator has no effect}} expected-warning {{unused}}

  // This is array indexing not an array declaration because a braced-init-list
  // is not syntactically a constant-expression.
  int(x[{0}]); // expected-error {{array subscript is not an integer}}
  struct A {
    struct Q { int n; };
    int operator[](Q);
  } a;
  int(a[{0}]); // expected-warning {{unused}}

  // These are array declarations.
  int(x[((void)1,1)]); // expected-error {{redefinition}}
  int(x[true ? 1 : (1,1)]); // expected-error {{redefinition}} // expected-warning {{left operand of comma operator has no effect}}

  int (*_Atomic atomic_ptr_to_int);
  *atomic_ptr_to_int = 42;
}

namespace function_with_trailing {
struct Foo {
  Foo(int);
};
template <typename T> void bar()
  { Foo _(T::method()->mem()); }
}
