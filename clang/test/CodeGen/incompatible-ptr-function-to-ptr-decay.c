// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

// Issue 182534
int a();

// CHECK: error: passing 'int (*)()' to parameter of type '__global int (*)' changes address space of pointer
void b(__attribute__((opencl_global)) int(*) );

void c() {
  b(a);
}

