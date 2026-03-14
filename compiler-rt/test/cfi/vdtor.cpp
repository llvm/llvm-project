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

// Tests that the CFI enforcement also applies to virtual destructor calls made
// via 'delete'.

// REQUIRES: cxxabi

#include <stdio.h>
#include "utils.h"

struct A {
  virtual ~A();
};

A::~A() {}

struct B {
  virtual ~B();
};

B::~B() {}

int main() {
  create_derivers<B>();

  A *a = new A;
  break_optimization(a);

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  // CFI-DIAG: runtime error: control flow integrity check for type 'B' failed during virtual call
  // CFI-DIAG-NEXT: note: vtable is of type '{{(struct )?}}A'
  delete (B *)a; // UB here

  // CFI-NOT: {{^2$}}
  // NCFI: {{^2$}}
  fprintf(stderr, "2\n");
}
