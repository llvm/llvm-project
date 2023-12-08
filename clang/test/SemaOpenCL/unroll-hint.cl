//RUN: %clang_cc1 -O0 -cl-std=CL2.0 -fsyntax-only -verify %s

kernel void C (global int *x) {
  int I = 3;
  __attribute__((opencl_unroll_hint(I))) // expected-error {{'opencl_unroll_hint' attribute requires an integer constant}}
  while (I--);
}

kernel void E() {
  __attribute__((opencl_unroll_hint(2,4))) // expected-error {{'opencl_unroll_hint' attribute takes no more than 1 argument}}
  for(int i=0; i<100; i++);
}

kernel void F() {
  __attribute__((opencl_unroll_hint(-1))) // expected-error {{'opencl_unroll_hint' attribute requires a positive integral compile time constant expression}}
  for(int i=0; i<100; i++);
}
