// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s


int foo1(int case, int throw, int y) { // expected-error {{invalid parameter name: 'case' is a keyword}} \
                                          expected-error {{invalid}}
  // Trailing parameters should be recovered.
  y = 1;
}

int foo2(int case = 1); // expected-error {{invalid parameter}}
int foo3(int const); // ok: without parameter name.
// ok: override has special meaning when used after method functions. it can be
// used as name.
int foo4(int override);
int foo5(int x const); // expected-error {{expected ')'}} expected-note {{to match this '('}}
// FIXME: bad recovery on the case below, "invalid parameter" is desired, the
// followon diagnostics should be suppressed.
int foo6(int case __attribute((weak))); // expected-error {{invalid parameter}}  \
                                        // expected-error {{expected ')'}} expected-note {{to match this '('}}

void test() {
  // FIXME: we shoud improve the dianostics for the following cases.
  int case; // expected-error {{expected unqualified-id}}
  struct X {
    int case; // expected-error {{expected member name or ';'}}
  };
}
