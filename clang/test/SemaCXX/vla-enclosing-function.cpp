// RUN: %clang_cc1 -fsyntax-only -Wno-vla-cxx-extension -verify %s

// GH55686
int foo();

void ctor() {
  using X = int[foo()]; // expected-note {{size expression is evaluated here}}
  struct S { S() { X x; } }; // expected-error {{cannot be used in a local class because its size is evaluated in an enclosing function}}
}

void sizeof_enclosing_vla_var() {
  int a[foo()]; // expected-note {{'a' declared here}}
  struct S { unsigned long f() { return sizeof(a); } }; // expected-error {{reference to local variable 'a' declared in enclosing function}}
}

void decltype_var() {
  int a[foo()]; // expected-note {{size expression is evaluated here}}
  struct S { void f() { decltype(a) b; } }; // expected-error {{cannot be used in a local class}}
}

void nested_local_class() {
  using X = int[foo()]; // expected-note {{size expression is evaluated here}}
  struct S {
    void f() { struct T { T() { X x; } }; } // expected-error {{cannot be used in a local class}}
  };
}

void pointer_to_vla() {
  using P = int (*)[foo()]; // expected-note {{size expression is evaluated here}}
  struct S { unsigned long f(P p) { return sizeof(*p); } }; // expected-error {{cannot be used in a local class}}
}

template <typename T> void in_template() {
  using X = int[foo()]; // expected-note {{size expression is evaluated here}}
  struct S { S() { X x; } }; // expected-error {{cannot be used in a local class}}
}

namespace GH55686 {
void f(int n) {
  using T = int[n]; // expected-note 2 {{size expression is evaluated here}}
  struct A {
    using U = T; // expected-error {{cannot be used in a local class}}
    void f() { U u; }
    void g() { T t; } // expected-error {{cannot be used in a local class}}
  };
  A::U u;
  A a;
  a.f();
  a.g();
}
} // namespace GH55686

// Valid: no diagnostics.
int same_function() {
  using X = int[foo()];
  X x;
  return sizeof(X) + sizeof(x);
}

void own_vla_in_local_class() {
  struct S { void f() { int a[foo()]; (void)sizeof(a); } };
}

void param_vla_in_local_class() {
  struct S { unsigned long f(int n, int (*p)[n]) { return sizeof(*p); } };
}

int lambda_sizeof() {
  using X = int[foo()];
  return [&] { return (int)sizeof(X); }();
}
