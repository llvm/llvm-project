// RUN: %clang_cc1 -fsyntax-only -verify %s 

// Issue 182534
int a();

void b(__attribute__((opencl_global)) int(*) );
// expected-note@-1 {{passing argument to parameter here}}

void c() {
  b(a);
  // expected-error@-1 {{passing 'int (*)()' to parameter of type '__global int (*)' changes address space of pointer}}
}

