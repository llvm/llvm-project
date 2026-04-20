// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -verify %s 

// Issue 182534
int foo();

void bar(__attribute__((opencl_global)) int*); // #cldecl
void baz(__ob_wrap int*); // #ofdecl

void a() {
  bar(foo);
  // expected-error@-1 {{passing 'int (*)()' to parameter of type '__global int *' changes address space of pointer}}
  // expected-note@#cldecl {{passing argument to parameter here}}
  __ob_trap int val[10];
  baz(val);
  // expected-error@-1 {{assigning to '__ob_wrap int *' from '__ob_trap int *' with incompatible overflow behavior types ('__ob_wrap' and '__ob_trap')}}
  // expected-note@#ofdecl {{passing argument to parameter here}}
}

