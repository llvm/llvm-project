// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

void __attribute__((overloadable)) foo(event_t, __local char *); // expected-note {{candidate function}}
void __attribute__((overloadable)) foo(event_t, __local float *); // expected-note {{candidate function}}

void kernel ker(__local char *src1, __local float *src2, __global int *src3) {
  event_t evt;
  foo(evt, src1);
  foo(0, src2);
  foo(evt, src3); // expected-error {{no matching function for call to 'foo'}}
}
