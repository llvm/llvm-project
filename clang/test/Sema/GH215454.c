// RUN: %clang_cc1 -std=gnu99 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=gnu99 -fsyntax-only -verify -fno-recovery-ast %s

// A declaration that declares nothing made the enclosing statement expression
// invalid without an error, so the void condition below was never diagnosed
// and CodeGen crashed on it.

void d2(int e) {
  if (({ ; __typeof__(e); })) {} // expected-warning {{declaration does not declare anything}} \
                                 // expected-error {{statement requires expression of scalar type ('void' invalid)}}
}

// In C a declaration is not a statement, so this never reaches the fixed code.
void d3(void) {
  if (1)
    int; // expected-error {{expected expression}}
}

// The statement expression has type void, not the type of '1'. The failing
// assertion checks that it is evaluated at all; it used to be dropped.
#define IS_SAME(LHS, RHS) _Generic(typeof(LHS), RHS : 1, default : 0)
void d4(void) {
  _Static_assert(IS_SAME(({1; int;}), void), ""); // expected-warning {{declaration does not declare anything}}
  _Static_assert(IS_SAME(({1; int;}), int), "");  // expected-warning {{declaration does not declare anything}} \
                                                   // expected-error {{static assertion failed}}
}

// Reproducer from the issue; the unclosed '({' makes recovery run to EOF.
#define c(a, b)                                                                \
  {;__typeof__(b);}
void d(int e) {if((c(, e););  // expected-warning {{'(' and '{' tokens introducing statement expression appear in different macro expansion contexts}} \
                              // expected-note {{'{' token is here}} \
                              // expected-warning {{declaration does not declare anything}} \
                              // expected-error {{unexpected ';' before ')'}} \
                              // expected-note {{to match this '{'}}
}                             // expected-error {{expected expression}} \
                              // expected-error@+1 {{expected '}'}}
