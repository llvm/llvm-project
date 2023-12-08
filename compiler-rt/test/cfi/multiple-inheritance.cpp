// RUN: %clangxx_cfi -o %t1 %s
// RUN: %expect_crash %run %t1 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %run %t1 x 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB32 -o %t2 %s
// RUN: %expect_crash %run %t2 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %run %t2 x 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB64 -o %t3 %s
// RUN: %expect_crash %run %t3 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %run %t3 x 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DBM -o %t4 %s
// RUN: %expect_crash %run %t4 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %run %t4 x 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -o %t5 %s
// RUN: %run %t5 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %run %t5 x 2>&1 | FileCheck --check-prefix=NCFI %s

// RUN: %clangxx_cfi_diag -o %t6 %s
// RUN: %run %t6 2>&1 | FileCheck --check-prefix=CFI-DIAG2 %s
// RUN: %run %t6 x 2>&1 | FileCheck --check-prefix=CFI-DIAG1 %s

// Tests that the CFI mechanism is sensitive to multiple inheritance and only
// permits calls via virtual tables for the correct base class.

// REQUIRES: cxxabi

#include <stdio.h>
#include "utils.h"

struct A {
  virtual void f() = 0;
};

struct B {
  virtual void g() = 0;
};

struct C : A, B {
  virtual void f(), g();
};

void C::f() {}
void C::g() {}

int main(int argc, char **argv) {
  create_derivers<A>();
  create_derivers<B>();

  C *c = new C;
  break_optimization(c);

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  if (argc > 1) {
    A *a = c;
    // CFI-DIAG1: runtime error: control flow integrity check for type 'B' failed during cast to unrelated type
    // CFI-DIAG1-NEXT: note: vtable is of type '{{(struct )?}}C'
    ((B *)a)->g(); // UB here
  } else {
    // CFI-DIAG2: runtime error: control flow integrity check for type 'A' failed during cast to unrelated type
    // CFI-DIAG2-NEXT: note: vtable is of type '{{(struct )?}}C'
    B *b = c;
    ((A *)b)->f(); // UB here
  }

  // CFI-NOT: {{^2$}}
  // NCFI: {{^2$}}
  fprintf(stderr, "2\n");
}
