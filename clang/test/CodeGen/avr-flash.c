// RUN: %clang_cc1 -triple avr -emit-llvm-only -verify %s

int foo(void) {
  static __flash int b[] = {4, 6}; // expected-error {{qualifier 'const' is needed for variables in address space '__flash'}}
  return b[0];
}
