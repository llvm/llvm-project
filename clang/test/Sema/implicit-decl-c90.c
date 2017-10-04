// RUN: %clang_cc1 %s -std=c90 -verify -fsyntax-only
void t0(int x) {
  int explicit_decl();
  int (*p)();
  if(x > 0)
    x = g() + 1; // expected-note {{previous implicit declaration}}
  p = g;
  if(x < 0) {
    extern void u(int (*)[h()]);
    int (*q)() = h;
  }
  p = h; /* expected-error {{use of undeclared identifier 'h'}} */
}

void t1(int x) {
  int (*p)();
  switch (x) {
    g();
  case 0:
    x = h() + 1;
    break;
  case 1:
    p = g;
    p = h;
    break;
  }
  p = g; /* expected-error {{use of undeclared identifier 'g'}} */
  p = h; /* expected-error {{use of undeclared identifier 'h'}} */
  explicit_decl();
  p = explicit_decl;
}

int t2(int x) {
  int y = ({ if (x > 0) x = g() + 1; 2*x; });
  int (*p)() = g; /* expected-error {{use of undeclared identifier 'g'}} */
  return y;
}

int PR34822() {
  {int i = sizeof(PR34822_foo());} /* expected-note {{previous definition is here}} */
  {extern int PR34822_foo;} /* expected-error {{redefinition of 'PR34822_foo' as different kind of symbol}} */

  {extern int PR34822_bar;} /* expected-note {{previous declaration is here}} */
  {int i = sizeof(PR34822_bar());} /* expected-warning {{use of out-of-scope declaration of 'PR34822_bar' whose type is not compatible with that of an implicit declaration}} expected-error {{called object type 'int' is not a function or function pointer}} */
}

int (*p)() = g; /* expected-error {{use of undeclared identifier 'g'}} */
int (*q)() = h; /* expected-error {{use of undeclared identifier 'h'}} */

float g(); /* expected-error {{conflicting types for 'g'}} */
