// RUN: %clang_cc1 -verify %s

void f1() {
  int a = 1;
  int b = __imag a;
  int *c = &__real a;
  int *d = &__imag a; // expected-error {{must be an lvalue}}
}

void f2() {
  _Complex int a = 1;
  int b = __imag a;
  int *c = &__real a;
  int *d = &__imag a;
}

void f3() {
  double a = 1;
  double b = __imag a;
  double *c = &__real a;
  double *d = &__imag a; // expected-error {{must be an lvalue}}
}

void f4() {
  _Complex double a = 1;
  double b = __imag a;
  double *c = &__real a;
  double *d = &__imag a;
}
