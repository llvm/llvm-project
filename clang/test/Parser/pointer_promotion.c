// RUN: %clang_cc1 -fsyntax-only -verify %s

void test() {
  void *vp;
  int *ip;
  char *cp;
  struct foo *fp;
  struct bar *bp;
  short sint = 7;

  if (ip < cp) {} // expected-warning {{comparison of distinct pointer types ('int *' and 'char *')}}
  if (cp < fp) {} // expected-warning {{comparison of distinct pointer types ('char *' and 'struct foo *')}}
  if (fp < bp) {} // expected-warning {{comparison of distinct pointer types ('struct foo *' and 'struct bar *')}}
  if (ip < 7) {} // expected-warning {{comparison between pointer and integer ('int *' and 'int')}}
  if (sint < ip) {} // expected-warning {{comparison between pointer and integer ('short' and 'int *')}}
  if (ip == cp) {} // expected-warning {{comparison of distinct pointer types ('int *' and 'char *')}}
}
