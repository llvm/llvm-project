// RUN: %clang_cc1 -triple x86_64-linux -target-feature -sse -target-feature -sse2 -S -o /dev/null -verify %s
// REQUIRES: x86-registered-target

double f1(void) { // expected-error {{SSE register return with SSE disabled}}
  return 1.4;
}
extern double g;
void f2(void) { // expected-error {{SSE register return with SSE disabled}}
  g = f1();
}
void take_double(double);
void pass_double(void) {
  // FIXME: Still asserts.
  //take_double(1.5);
}
