// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -fsyntax-only -fblocks -pedantic -verify %s

void cat0(int a[static 0]) {} // expected-warning {{zero size arrays are an extension}} \
                              // expected-note {{callee declares array parameter as static here}}

void cat(int a[static 3]) {} // expected-note 4 {{callee declares array parameter as static here}} expected-note 2 {{passing argument to parameter 'a' here}}

void vat(int i, int a[static i]) {} // expected-note {{callee declares array parameter as static here}}

void f(int *p) {
  int a[2], b[3], c[4];

  cat0(0); // expected-warning {{null passed to a callee that requires a non-null argument}}

  cat(0); // expected-warning {{null passed to a callee that requires a non-null argument}}
  cat(a); // expected-warning {{array argument is too small; contains 2 elements, callee requires at least 3}}
  cat(b);
  cat(c);
  cat(p);

  vat(1, 0); // expected-warning {{null passed to a callee that requires a non-null argument}}
  vat(3, b);

  char d[4];
  cat((int *)d); // expected-warning {{array argument is too small; is of size 4, callee requires at least 12}}
  cat(d); // expected-warning {{array argument is too small; is of size 4, callee requires at least 12}} expected-warning {{incompatible pointer types}}

  char e[12];
  cat((int *)e);
  cat(e); // expected-warning {{incompatible pointer types}}
}


typedef int td[static 3]; // expected-error {{'static' used in array declarator outside of function prototype}}
typedef void(*fp)(int[static 42]); // no-warning

void g(void) {
  int a[static 42]; // expected-error {{'static' used in array declarator outside of function prototype}}

  int b[const 10]; // expected-error {{type qualifier used in array declarator outside of function prototype}}
  int c[volatile 10]; // expected-error {{type qualifier used in array declarator outside of function prototype}}
  int d[restrict 10]; // expected-error {{type qualifier used in array declarator outside of function prototype}}

  int e[static restrict 1]; // expected-error {{'static' used in array declarator outside of function prototype}}
}

void h(int [static const 10][42]); // no-warning

void i(int [10]
       [static 42]); // expected-error {{'static' used in non-outermost array type derivation}}

void j(int [10]
       [const 42]); // expected-error {{type qualifier used in non-outermost array type derivation}}

void k(int (*x)[static 10]); // expected-error {{'static' used in non-outermost array type derivation}}
void l(int (x)[static 10]); // no-warning
void m(int *x[static 10]); // no-warning
void n(int *(x)[static 10]); // no-warning

void o(int (x[static 10])(void)); // expected-error{{'x' declared as array of functions of type 'int (void)'}}
void p(int (^x)[static 10]); // expected-error{{block pointer to non-function type is invalid}}
void q(int (^x[static 10])()); // no-warning

void r(x)
  int x[restrict]; // no-warning
{}
