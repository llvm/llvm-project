// RUN: %clangxx_cfi -o %t1 %s
// RUN: %expect_crash %run %t1 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB32 -o %t2 %s
// RUN: %expect_crash %run %t2 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB64 -o %t3 %s
// RUN: %expect_crash %run %t3 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DBM -o %t4 %s
// RUN: %expect_crash %run %t4 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -o %t5 %s
// RUN: %run %t5 2>&1 | FileCheck --check-prefix=NCFI %s

// RUN: %clangxx_cfi_diag -o %t6 %s
// RUN: %run %t6 2>&1 | FileCheck --check-prefix=CFI-DIAG %s

// Tests that the CFI mechanism crashes the program when making a non-virtual
// call to an object of the wrong class, by casting a pointer to such an object
// and attempting to make a call through it.

// REQUIRES: cxxabi

#include <stdio.h>
#include "utils.h"

struct A {
  virtual void v();
};

void A::v() {}

struct B {
  void f();
  virtual void g();
};

void B::f() {}
void B::g() {}

int main() {
  create_derivers<B>();

  A *a = new A;
  break_optimization(a);

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  // CFI-DIAG: runtime error: control flow integrity check for type 'B' failed during non-virtual call
  // CFI-DIAG-NEXT: note: vtable is of type '{{(struct )?}}A'
  ((B *)a)->f(); // UB here

  // CFI-NOT: {{^2$}}
  // NCFI: {{^2$}}
  fprintf(stderr, "2\n");
}
