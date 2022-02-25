// RUN: %clang_cc1 -triple avr -target-cpu atxmega384c3 -emit-llvm-only -verify %s

int foo(void) {
  static __flash  int b[] = {4, 6}; // expected-error {{qualifier 'const' is needed for variables in address space '__flash*'}}
  static __flash1 int c[] = {8, 1}; // expected-error {{qualifier 'const' is needed for variables in address space '__flash*'}}
  static __flash2 int d[] = {8, 1}; // expected-error {{qualifier 'const' is needed for variables in address space '__flash*'}}
  static __flash3 int e[] = {8, 1}; // expected-error {{qualifier 'const' is needed for variables in address space '__flash*'}}
  static __flash4 int f[] = {8, 1}; // expected-error {{qualifier 'const' is needed for variables in address space '__flash*'}}
  static __flash5 int g[] = {8, 1}; // expected-error {{qualifier 'const' is needed for variables in address space '__flash*'}}
  return b[0] + c[1];
}
