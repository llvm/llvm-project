// RUN: %clang_cc1 -triple x86_64-linux -target-feature -sse -target-feature -sse2 -S -o /dev/null -verify %s
// REQUIRES: x86-registered-target

// expected-error@+2{{SSE register return with SSE disabled}}
// expected-note@+1{{'f1' defined here}}
double f1(void) {
  return 1.4;
}
extern double g;
void f2(void) {
  g = f1();
}
void take_double(double);
void pass_double(void) {
  take_double(1.5);
}
