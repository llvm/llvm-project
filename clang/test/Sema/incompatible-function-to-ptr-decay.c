// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -verify %s 

// Issue 182534
int foo();

void bar(__attribute__((opencl_global)) int*); // #cldecl
void baz(__ob_wrap int*); // #ofdecl

void a() {
  bar(foo);
  __ob_trap int val[10];
  baz(val);
  // expected-error@-1 {{passing 'int (*)()' to parameter of type '__ob_wrap int (*)' changes address space of pointer}}
  // expected-note@#ofdecl {{passing argument to parameter here}}
}

