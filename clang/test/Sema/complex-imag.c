// RUN: %clang_cc1 -verify %s

void f1(void) {
  int a = 1;
  int b = __imag a;
  int *c = &__real a;
  int *d = &__imag a; // expected-error {{cannot take the address of an rvalue of type 'int'}}
}

void f2(void) {
  _Complex int a = 1;
  int b = __imag a;
  int *c = &__real a;
  int *d = &__imag a;
}

void f3(void) {
  double a = 1;
  double b = __imag a;
  double *c = &__real a;
  double *d = &__imag a; // expected-error {{cannot take the address of an rvalue of type 'double'}}
}

void f4(void) {
  _Complex double a = 1;
  double b = __imag a;
  double *c = &__real a;
  double *d = &__imag a;
}
