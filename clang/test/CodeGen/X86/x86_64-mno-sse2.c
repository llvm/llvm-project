// RUN: %clang_cc1 -triple x86_64-linux -target-feature -sse2 -S -o /dev/null -verify %s
// REQUIRES: x86-registered-target

double f1(void) { // expected-error {{SSE2 register return with SSE2 disabled}}
  return 1.4;
}
extern double g;
void f2(void) { // expected-error {{SSE2 register return with SSE2 disabled}}
  g = f1();
}
void take_double(double);
void pass_double(void) {
  // FIXME: Still asserts.
  //take_double(1.5);
}

double return_double();
void call_double(double *a) { // expected-error {{SSE2 register return with SSE2 disabled}}
  *a = return_double();
}
