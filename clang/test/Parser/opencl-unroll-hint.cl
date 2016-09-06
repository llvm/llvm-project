//RUN: %clang_cc1 -O0 -cl-std=CL2.0 -fsyntax-only -verify %s

kernel void B (global int *x) {
  __attribute__((opencl_unroll_hint(42)))
  if (x[0])                             // expected-error {{OpenCL only supports 'opencl_unroll_hint' attribute on for, while, and do statements}}
    x[0] = 15;
}

