// RUN: %clangxx_cfi -o %t1 %s
// RUN: %expect_crash_unless_devirt %t1 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB32 -o %t2 %s
// RUN: %expect_crash %t2 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB64 -o %t3 %s
// RUN: %expect_crash %t3 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DBM -o %t4 %s
// RUN: %expect_crash %t4 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -o %t5 %s
// RUN: %t5 2>&1 | FileCheck --check-prefix=NCFI %s

// RUN: %clangxx_cfi_diag -o %t6 %s
// RUN: %t6 2>&1 | FileCheck --check-prefix=CFI-DIAG %s

// Tests that the CFI mechanism crashes the program when a virtual table is
// replaced with a compatible table of function pointers that does not belong to
// any class, by manually overwriting the virtual table of an object and
// attempting to make a call through it.

// REQUIRES: cxxabi

#include <stdio.h>
#include "utils.h"

struct A {
  virtual void f();
};

void A::f() {}

void foo() {
  fprintf(stderr, "foo\n");
}

void *fake_vtable[] = { 0, 0, (void *)&foo };

int main() {
  create_derivers<A>();

  A *a = new A;
  *((void **)a) = fake_vtable + 2; // UB here
  break_optimization(a);

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  // CFI-NOT: foo
  // NCFI: foo
  // CFI-DIAG: runtime error: control flow integrity check for type 'A' failed during virtual call
  // CFI-DIAG-NEXT: note: invalid vtable
  a->f();

  // We don't check for the absence of a 2 here because under devirtualization
  // our virtual call may be devirtualized and we will proceed with execution
  // rather than crashing.

  // NCFI: {{^2$}}
  fprintf(stderr, "2\n");
}
