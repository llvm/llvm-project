// RUN: %clangxx_cfi -o %t1 %s
// RUN: %expect_crash %t1 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB32 -o %t2 %s
// RUN: %expect_crash %t2 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB64 -o %t3 %s
// RUN: %expect_crash %t3 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DBM -o %t4 %s
// RUN: %expect_crash %t4 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -o %t5 %s
// RUN: %expect_crash %t5 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -DB32 -o %t6 %s
// RUN: %expect_crash %t6 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -DB64 -o %t7 %s
// RUN: %expect_crash %t7 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -DBM -o %t8 %s
// RUN: %expect_crash %t8 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -o %t9 %s
// RUN: %expect_crash %t9 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -DB32 -o %t10 %s
// RUN: %expect_crash %t10 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -DB64 -o %t11 %s
// RUN: %expect_crash %t11 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -DBM -o %t12 %s
// RUN: %expect_crash %t12 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -o %t13 %s
// RUN: %expect_crash %t13 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -DB32 -o %t14 %s
// RUN: %expect_crash %t14 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -DB64 -o %t15 %s
// RUN: %expect_crash %t15 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -DBM -o %t16 %s
// RUN: %expect_crash %t16 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_diag -o %t17 %s
// RUN: %t17 2>&1 | FileCheck --check-prefix=CFI-DIAG %s

// RUN: %clangxx -o %t18 %s
// RUN: %t18 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism crashes the program when making a
// base-to-derived cast from a destructor of the base class,
// where both types have virtual tables.

// REQUIRES: cxxabi

#include <stdio.h>
#include "utils.h"

template<typename T>
class A {
 public:
  T* context() { return static_cast<T*>(this); }

  virtual ~A() {
    break_optimization(context());
  }
};

class B : public A<B> {
 public:
  virtual ~B() { }
};

int main() {
  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  // CFI-DIAG: runtime error: control flow integrity check for type 'B' failed during base-to-derived cast
  // CFI-DIAG-NEXT: note: vtable is of type '{{(class )?}}A<{{(class )?}}B>'
  B* b = new B;
  break_optimization(b);
  delete b; // UB here

  // CFI-NOT: {{^2$}}
  // NCFI: {{^2$}}
  fprintf(stderr, "2\n");
}
