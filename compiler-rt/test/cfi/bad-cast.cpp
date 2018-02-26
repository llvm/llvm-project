// RUN: %clangxx_cfi -o %t1 %s
// RUN: %expect_crash %run %t1 a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t1 b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t1 c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %run %t1 d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t1 e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t1 f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %expect_crash %run %t1 g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %run %t1 h 2>&1 | FileCheck --check-prefix=PASS %s

// RUN: %clangxx_cfi -DB32 -o %t2 %s
// RUN: %expect_crash %run %t2 a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t2 b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t2 c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %run %t2 d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t2 e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t2 f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %expect_crash %run %t2 g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %run %t2 h 2>&1 | FileCheck --check-prefix=PASS %s

// RUN: %clangxx_cfi -DB64 -o %t3 %s
// RUN: %expect_crash %run %t3 a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t3 b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t3 c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %run %t3 d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t3 e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t3 f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %expect_crash %run %t3 g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %run %t3 h 2>&1 | FileCheck --check-prefix=PASS %s

// RUN: %clangxx_cfi -DBM -o %t4 %s
// RUN: %expect_crash %run %t4 a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t4 b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t4 c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %run %t4 d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t4 e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t4 f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %expect_crash %run %t4 g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %run %t4 h 2>&1 | FileCheck --check-prefix=PASS %s

// RUN: %clangxx_cfi -fsanitize=cfi-cast-strict -o %t5 %s
// RUN: %expect_crash %run %t5 a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t5 b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t5 c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t5 d 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t5 e 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t5 f 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t5 g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %expect_crash %run %t5 h 2>&1 | FileCheck --check-prefix=FAIL %s

// RUN: %clangxx -o %t6 %s
// RUN: %run %t6 a 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t6 b 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t6 c 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t6 d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t6 e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t6 f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t6 g 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %run %t6 h 2>&1 | FileCheck --check-prefix=PASS %s

// RUN: %clangxx_cfi_diag -o %t7 %s
// RUN: %run %t7 a 2>&1 | FileCheck --check-prefix=CFI-DIAG-D %s
// RUN: %run %t7 b 2>&1 | FileCheck --check-prefix=CFI-DIAG-D %s
// RUN: %run %t7 c 2>&1 | FileCheck --check-prefix=CFI-DIAG-D %s
// RUN: %run %t7 g 2>&1 | FileCheck --check-prefix=CFI-DIAG-U %s

// Tests that the CFI enforcement detects bad casts.

// REQUIRES: cxxabi

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

int main(int argc, char **argv) {
  create_derivers<B>();

  B *b = new B;
  break_optimization(b);

  // FAIL: 1
  // PASS: 1
  fprintf(stderr, "1\n");

  A a;

  // CFI-DIAG-D: runtime error: control flow integrity check for type 'B' failed during base-to-derived cast
  // CFI-DIAG-D-NEXT: note: vtable is of type '{{(struct )?}}A'

  // CFI-DIAG-U: runtime error: control flow integrity check for type 'B' failed during cast to unrelated type
  // CFI-DIAG-U-NEXT: note: vtable is of type '{{(struct )?}}A'

  switch (argv[1][0]) {
    case 'a':
      static_cast<B *>(&a); // UB
      break;
    case 'b':
      static_cast<B &>(a); // UB
      break;
    case 'c':
      static_cast<B &&>(a); // UB
      break;
    case 'd':
      static_cast<C *>(&a); // UB, strict only
      break;
    case 'e':
      static_cast<C &>(a); // UB, strict only
      break;
    case 'f':
      static_cast<C &&>(a); // UB, strict only
      break;
    case 'g':
      static_cast<B *>(static_cast<void *>(&a)); // Non-UB bad cast
      break;
    case 'h':
      static_cast<C *>(static_cast<void *>(&a)); // Non-UB bad cast, strict only
      break;
  }

  // FAIL-NOT: {{^2$}}
  // PASS: {{^2$}}
  fprintf(stderr, "2\n");
}
