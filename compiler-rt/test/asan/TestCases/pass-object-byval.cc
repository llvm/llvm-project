// Verify that objects passed by value get red zones and that the copy
// constructor is called.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --implicit-check-not \
// RUN:     Assertion{{.*}}failed

// ASan instrumentation can't insert red-zones around inalloca parameters.
// XFAIL: win32 && asan-32-bits

#include <cassert>

class A {
 public:
  A() : me(this) {}
  A(const A &other) : me(this) {
    for (int i = 0; i < 8; ++i) a[i] = other.a[i];
  }

  int a[8];
  A *me;
};

int bar(A *a) {
  int *volatile ptr = &a->a[0];
  return *(ptr - 1);
}

void foo(A a) {
  assert(a.me == &a);
  bar(&a);
}

int main() {
  A a;
  foo(a);
}

// CHECK: ERROR: AddressSanitizer: stack-buffer-overflow
// CHECK: READ of size 4 at
// CHECK: is located in stack of thread
