//RUN: %clang_cc1 -cl-std=CL2.0 -fsyntax-only -verify %s

kernel void B (global int *x) {
  __attribute__((opencl_unroll_hint(42))) // expected-error {{'opencl_unroll_hint' attribute only applies to 'for', 'while', and 'do' statements}}
  if (x[0])
    x[0] = 15;
}

void parse_order_error() {
  // Ensure we properly diagnose OpenCL loop attributes on the incorrect
  // subject in the presence of other attributes.
  int i = 1000;
  __attribute__((nomerge, opencl_unroll_hint(8))) // expected-error {{'opencl_unroll_hint' attribute only applies to 'for', 'while', and 'do' statements}}
  if (i) { parse_order_error(); } // Recursive call silences unrelated diagnostic about nomerge.

  __attribute__((opencl_unroll_hint(8), nomerge)) // expected-error {{'opencl_unroll_hint' attribute only applies to 'for', 'while', and 'do' statements}}
  if (i) { parse_order_error(); } // Recursive call silences unrelated diagnostic about nomerge.

  __attribute__((nomerge, opencl_unroll_hint(8))) // OK
  while (1) { parse_order_error(); } // Recursive call silences unrelated diagnostic about nomerge.
}
