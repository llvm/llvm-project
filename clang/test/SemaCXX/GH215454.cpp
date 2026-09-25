// RUN: %clang_cc1 -fsyntax-only -verify %s

// A declaration that declares nothing is still diagnosed as before; the
// enclosing 'if' is now kept with a null statement as its body instead of
// being dropped.
void foo();
void f() {
  if (1)
    int; // expected-warning {{declaration does not declare anything}}

  if (foo(), 1)
    int; // expected-warning {{declaration does not declare anything}}

  if (1) int; // expected-warning {{declaration does not declare anything}} \
              // expected-warning {{if statement has empty body}} \
              // expected-note {{put the semicolon on a separate line to silence this warning}}
}
