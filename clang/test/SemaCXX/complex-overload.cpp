// RUN: %clang_cc1 -fsyntax-only -verify %s
char *foo(float);

void test_foo_1(float fv, double dv, float _Complex fc, double _Complex dc) {
  char *cp1 = foo(fv);
  char *cp2 = foo(dv);
  // Note: GCC and EDG reject these two, they are valid C99 conversions but
  // shouldn't be accepted in C++ because the result is surprising.
  char *cp3 = foo(fc); // expected-error {{implicit conversion from '_Complex float' to 'float' is not permitted in C++}}
  char *cp4 = foo(dc); // expected-error {{implicit conversion from '_Complex double' to 'float' is not permitted in C++}}
}

int *foo(float _Complex);

void test_foo_2(float fv, double dv, float _Complex fc, double _Complex dc) {
  char *cp1 = foo(fv);
  char *cp2 = foo(dv);
  int *ip = foo(fc);
  int *lp = foo(dc);
}

long *foo(double _Complex);

void test_foo_3(float fv, double dv, float _Complex fc, double _Complex dc) {
  char *cp1 = foo(fv);
  char *cp2 = foo(dv);
  int *ip = foo(fc);
  long *lp = foo(dc);
}

char *promote_or_convert(double _Complex);  // expected-note{{candidate function}}
int *promote_or_convert(long double _Complex); // expected-note{{candidate function}} 

void test_promote_or_convert(float f, float _Complex fc) {
  char *cp = promote_or_convert(fc);
  int *ip2 = promote_or_convert(f); // expected-error{{call to 'promote_or_convert' is ambiguous}}
}

char *promote_or_convert2(float);
int *promote_or_convert2(double _Complex);

void test_promote_or_convert2(float _Complex fc) {
  int *cp = promote_or_convert2(fc);
}

char *promote_or_convert3(int _Complex); // expected-note {{candidate}}
int *promote_or_convert3(long _Complex); // expected-note {{candidate}}

void test_promote_or_convert3(short _Complex sc) {
  char *cp1 = promote_or_convert3(sc);
  char *cp2 = promote_or_convert3(1i);
  int *cp3 = promote_or_convert3(1il);
  int *cp4 = promote_or_convert3(1ill); // expected-error {{ambiguous}}
}

char &convert4(short _Complex);
char &test_convert4 = convert4(1i);
