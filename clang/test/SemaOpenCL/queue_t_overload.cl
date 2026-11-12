// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -pedantic -fsyntax-only

void __attribute__((overloadable)) foo(queue_t, __local char *); // expected-note {{candidate function not viable: no known conversion from 'int' to '__private queue_t' for 1st argument}} // expected-note {{candidate function}}
void __attribute__((overloadable)) foo(queue_t, __local float *); // expected-note {{candidate function not viable: no known conversion from 'int' to '__private queue_t' for 1st argument}} // expected-note {{candidate function}}

void kernel ker(__local char *src1, __local float *src2, __global int *src3) {
  queue_t q;
  foo(q, src1);
  foo(0, src2);
  foo(q, src3); // expected-error {{no matching function for call to 'foo'}}
  foo(1, src3); // expected-error {{no matching function for call to 'foo'}}
}
