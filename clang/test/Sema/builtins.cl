// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic

kernel void test(global float *out, global float *in, global int* in2) {
  out[0] = __builtin_nanf("");
  __builtin_memcpy(out, in, 32);
  out[0] = __builtin_frexpf(in[0], in2);
}

void pr28651(void) {
  __builtin_alloca(value); // expected-error{{use of undeclared identifier}}
}
