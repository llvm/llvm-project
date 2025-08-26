// RUN: %clangxx_cfi_dso -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso %s -o %t %ld_flags_rpath_exe
// RUN: %expect_crash %t 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t x 2>&1 | FileCheck --check-prefix=CFI-CAST %s

// RUN: %clangxx_cfi_dso -DB32 -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso -DB32 %s -o %t %ld_flags_rpath_exe
// RUN: %expect_crash %t 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t x 2>&1 | FileCheck --check-prefix=CFI-CAST %s

// RUN: %clangxx_cfi_dso -DB64 -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso -DB64 %s -o %t %ld_flags_rpath_exe
// RUN: %expect_crash %t 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t x 2>&1 | FileCheck --check-prefix=CFI-CAST %s

// RUN: %clangxx_cfi_dso -DBM -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso -DBM %s -o %t %ld_flags_rpath_exe
// RUN: %expect_crash %t 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t x 2>&1 | FileCheck --check-prefix=CFI-CAST %s

// RUN: %clangxx -DBM -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx -DBM %s -o %t %ld_flags_rpath_exe
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t x 2>&1 | FileCheck --check-prefix=NCFI %s

// RUN: %clangxx -DBM -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso -DBM %s -o %t %ld_flags_rpath_exe
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t x 2>&1 | FileCheck --check-prefix=NCFI %s

// RUN: %clangxx_cfi_dso_diag -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso_diag %s -o %t %ld_flags_rpath_exe
// RUN: %t 2>&1 | FileCheck --check-prefix=CFI-DIAG-CALL %s
// RUN: %t x 2>&1 | FileCheck --check-prefix=CFI-DIAG-CALL --check-prefix=CFI-DIAG-CAST %s

// Tests that the CFI mechanism crashes the program when making a virtual call
// to an object of the wrong class but with a compatible vtable, by casting a
// pointer to such an object and attempting to make a call through it.

// REQUIRES: cxxabi

#include <stdio.h>
#include <string.h>

struct A {
  virtual void f();
};

void *create_B();

#ifdef SHARED_LIB

#include "../utils.h"
struct B {
  virtual void f();
};
void B::f() {}

void *create_B() {
  create_derivers<B>();
  return (void *)(new B());
}

#else

void A::f() {}

int main(int argc, char *argv[]) {
  void *p = create_B();
  A *a;

  // CFI: =0=
  // CFI-CAST: =0=
  // NCFI: =0=
  fprintf(stderr, "=0=\n");

  if (argc > 1 && argv[1][0] == 'x') {
    // Test cast. BOOM.
    // CFI-DIAG-CAST: runtime error: control flow integrity check for type 'A' failed during cast to unrelated type
    // CFI-DIAG-CAST-NEXT: note: vtable is of type '{{(struct )?}}B'
    a = (A*)p;
  } else {
    // Invisible to CFI. Test virtual call later.
    memcpy(&a, &p, sizeof(a));
  }

  // CFI: =1=
  // CFI-CAST-NOT: =1=
  // NCFI: =1=
  fprintf(stderr, "=1=\n");

  // CFI-DIAG-CALL: runtime error: control flow integrity check for type 'A' failed during virtual call
  // CFI-DIAG-CALL-NEXT: note: vtable is of type '{{(struct )?}}B'
  a->f(); // UB here

  // CFI-NOT: =2=
  // CFI-CAST-NOT: =2=
  // NCFI: =2=
  fprintf(stderr, "=2=\n");
}
#endif
