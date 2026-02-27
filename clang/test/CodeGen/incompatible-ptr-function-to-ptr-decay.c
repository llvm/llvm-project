// RUN: %clang_cc1 -fsyntax-only -verify %s 

// Issue 182534
int a();

void b(__attribute__((opencl_global)) int(*) ); // #bdecl

void c() {
  b(a);
  // expected-error@-1 {{passing 'int (*)()' to parameter of type '__global int (*)' changes address space of pointer}}
  // expected-note@#bdecl {{passing argument to parameter here}}
}

