// RUN: %clang_cc1 -std=gnu99 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=gnu99 -fsyntax-only -verify -fno-recovery-ast %s

#define IS_SAME(LHS, RHS) _Generic(typeof(LHS), RHS : 1, default : 0)

void gh215454_type(void) {
  _Static_assert(IS_SAME(({1; int;}), void), ""); // expected-warning {{declaration does not declare anything}}
  _Static_assert(IS_SAME(({1; int;}), int), "");  // expected-warning {{declaration does not declare anything}} \
                                                   // expected-error {{static assertion failed}}
}

void gh215454_cond(int e) {
  if (({ ; __typeof__(e); })) {} // expected-warning {{declaration does not declare anything}} \
                                 // expected-error {{statement requires expression of scalar type ('void' invalid)}}
}

#define c(a, b)                                                                \
  {;__typeof__(b);}
void d(int e) {if((c(, e););  // expected-warning {{'(' and '{' tokens introducing statement expression appear in different macro expansion contexts}} \
                              // expected-note {{'{' token is here}} \
                              // expected-warning {{declaration does not declare anything}} \
                              // expected-error {{unexpected ';' before ')'}} \
                              // expected-note {{to match this '{'}}
}                             // expected-error {{expected expression}} \
                              // expected-error@+1 {{expected '}'}}
