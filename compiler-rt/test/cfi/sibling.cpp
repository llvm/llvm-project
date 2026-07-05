// XFAIL: *

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

// Tests that the CFI enforcement distinguishes between non-overriding siblings.
// XFAILed as not implemented yet.

#include <stdio.h>
#include "utils.h"

struct A {
  virtual void f();
};

void A::f() {}

struct B : A {
  virtual void f();
};

void B::f() {}

struct C : A {
};

int main() {
  create_derivers<B>();

  B *b = new B;
  break_optimization(b);

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  ((C *)b)->f(); // UB here

  // CFI-NOT: 2
  // NCFI: 2
  fprintf(stderr, "2\n");
}
