// RUN: %clang_cc1 -fsyntax-only -verify %s

typeof_unqual(int) u = 12; // expected-error {{expected function body after function declarator}}
__typeof_unqual(int) _u = 12;
__typeof_unqual__(int) __u = 12;

namespace GH97646 {
  template<bool B>
  void f() {
    __typeof__(B) x = false;
    !x; // expected-warning {{expression result unused}}
  }
}

// Ensure that __typeof_unqual / __typeof_unqual__ parse as a declaration
// specifier in block scope, for symmetry with __typeof__.
void block_scope_typeof_unqual() {
  __typeof_unqual(int) a = 0;
  __typeof_unqual__(int) b = 0;
  (void)a; (void)b;
}
