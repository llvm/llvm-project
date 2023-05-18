// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

struct A {
  int a[8];
};

int bar(A *a) {
  int *volatile ptr = &a->a[0];
  return *(ptr - 1);
}

void foo(A a) {
  bar(&a);
}

int main() {
  foo(A());
}

// CHECK: ERROR: AddressSanitizer: stack-buffer-underflow
// CHECK: READ of size 4 at
// CHECK: is located in stack of thread
