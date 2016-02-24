// RUN: %clang -o %t1 %s
// RUN: %t1 2>&1 | FileCheck --check-prefix=NCFI %s

// RUN: %clang_cfi -o %t2 %s
// RUN: %expect_crash %t2 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clang_cfi_diag -g -o %t3 %s
// RUN: %t3 2>&1 | FileCheck --check-prefix=CFI-DIAG %s

#include <stdio.h>

void f() {
}

int main() {
  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  // CFI-DIAG: runtime error: control flow integrity check for type 'void (int)' failed during indirect function call
  // CFI-DIAG: f defined here
  ((void (*)(int))f)(42); // UB here

  // CFI-NOT: 2
  // NCFI: 2
  fprintf(stderr, "2\n");
}
