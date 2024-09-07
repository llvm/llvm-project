// RUN: %clang_cc1 -fsyntax-only -verify %s

int *foo(__attribute__((opencl_private)) int *p,
         __attribute__((opencl_local)) int *l) {
  return p - l; // expected-error {{arithmetic operation with operands of type  ('__private int *' and '__local int *') which are pointers to non-overlapping address spaces}}
}
