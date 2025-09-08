// Cross-DSO diagnostics.
// The rules are:
// * If the library needs diagnostics, the main executable must request at
//   least some diagnostics as well (to link the diagnostic runtime).
// * -fsanitize-trap on the caller side overrides everything.
// * otherwise, the callee decides between trap/recover/norecover.

// Full-recover.
// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: %clangxx_cfi_dso_diag -g -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso_diag -g %s -o %t %ld_flags_rpath_exe

// RUN: %t icv 2>&1 | FileCheck %s --check-prefix=ICALL-DIAG --check-prefix=CAST-DIAG \
// RUN:                            --check-prefix=VCALL-DIAG --check-prefix=ALL-RECOVER

// RUN: %t i_v 2>&1 | FileCheck %s --check-prefix=ICALL-DIAG --check-prefix=CAST-NODIAG \
// RUN:                            --check-prefix=VCALL-DIAG --check-prefix=ALL-RECOVER

// RUN: %t _cv 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-DIAG \
// RUN:                            --check-prefix=VCALL-DIAG --check-prefix=ALL-RECOVER

// RUN: %t ic_ 2>&1 | FileCheck %s --check-prefix=ICALL-DIAG --check-prefix=CAST-DIAG \
// RUN:                            --check-prefix=VCALL-NODIAG --check-prefix=ALL-RECOVER

// Trap on icall, no-recover on cast.
// RUN: %clangxx_cfi_dso_diag -fsanitize-trap=cfi-icall -fno-sanitize-recover=cfi-unrelated-cast \
// RUN:     -g -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso_diag -fsanitize-trap=cfi-icall -fno-sanitize-recover=cfi-unrelated-cast \
// RUN:     -g %s -o %t %ld_flags_rpath_exe

// RUN: %expect_crash %t icv 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-NODIAG \
// RUN:                                          --check-prefix=VCALL-NODIAG --check-prefix=ICALL-FATAL

// RUN: not %t _cv 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-DIAG \
// RUN:                                --check-prefix=VCALL-NODIAG --check-prefix=CAST-FATAL

// RUN: %t __v 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-NODIAG \
// RUN:                            --check-prefix=VCALL-DIAG

// Callee: trap on icall, no-recover on cast.
// Caller: recover on everything.
// The same as in the previous case, behaviour is decided by the callee.
// RUN: %clangxx_cfi_dso_diag -fsanitize-trap=cfi-icall -fno-sanitize-recover=cfi-unrelated-cast \
// RUN:     -g -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso_diag \
// RUN:     -g %s -o %t %ld_flags_rpath_exe

// RUN: %expect_crash %t icv 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-NODIAG \
// RUN:                                          --check-prefix=VCALL-NODIAG --check-prefix=ICALL-FATAL

// RUN: not %t _cv 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-DIAG \
// RUN:                                --check-prefix=VCALL-NODIAG --check-prefix=CAST-FATAL

// RUN: %t __v 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-NODIAG \
// RUN:                            --check-prefix=VCALL-DIAG

// Caller in trapping mode, callee with full diagnostic+recover.
// Caller wins.
// cfi-nvcall is non-trapping in the main executable to link the diagnostic runtime library.
// RUN: %clangxx_cfi_dso_diag \
// RUN:     -g -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso -fno-sanitize-trap=cfi-nvcall \
// RUN:     -g %s -o %t %ld_flags_rpath_exe

// RUN: %expect_crash %t icv 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-NODIAG \
// RUN:                                          --check-prefix=VCALL-NODIAG --check-prefix=ICALL-FATAL

// RUN: %expect_crash %t _cv 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-NODIAG \
// RUN:                                          --check-prefix=VCALL-NODIAG --check-prefix=CAST-FATAL

// RUN: %expect_crash %t __v 2>&1 | FileCheck %s --check-prefix=ICALL-NODIAG --check-prefix=CAST-NODIAG \
// RUN:                                          --check-prefix=VCALL-NODIAG --check-prefix=VCALL-FATAL

// REQUIRES: cxxabi

#include <assert.h>
#include <stdio.h>
#include <string.h>

struct A {
  virtual void f();
};

void *create_B();

#ifdef SHARED_LIB

#include "../../utils.h"
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
  assert(argc == 2);
  assert(strlen(argv[1]) == 3);

  // ICALL-FATAL: =0=
  // CAST-FATAL:  =0=
  // VCALL-FATAL: =0=
  // ALL-RECOVER: =0=
  fprintf(stderr, "=0=\n");

  void *p;
  if (argv[1][0] == 'i') {
    // ICALL-DIAG: runtime error: control flow integrity check for type 'void *(int)' failed during indirect function call
    // ICALL-DIAG-NEXT: note: create_B() defined here
    // ICALL-NODIAG-NOT: runtime error: control flow integrity check {{.*}} during indirect function call
    p = ((void *(*)(int))create_B)(42);
  } else {
    p = create_B();
  }

  // ICALL-FATAL-NOT: =1=
  // CAST-FATAL:      =1=
  // VCALL-FATAL:     =1=
  // ALL-RECOVER:     =1=
  fprintf(stderr, "=1=\n");

  A *a;
  if (argv[1][1] == 'c') {
    // CAST-DIAG: runtime error: control flow integrity check for type 'A' failed during cast to unrelated type
    // CAST-DIAG-NEXT: note: vtable is of type '{{(struct )?}}B'
    // CAST-NODIAG-NOT: runtime error: control flow integrity check {{.*}} during cast to unrelated type
    a = (A*)p;
  } else {
    // Invisible to CFI.
    memcpy(&a, &p, sizeof(a));
  }

  // ICALL-FATAL-NOT: =2=
  // CAST-FATAL-NOT:  =2=
  // VCALL-FATAL:     =2=
  // ALL-RECOVER:     =2=
  fprintf(stderr, "=2=\n");

  // VCALL-DIAG: runtime error: control flow integrity check for type 'A' failed during virtual call
  // VCALL-DIAG-NEXT: note: vtable is of type '{{(struct )?}}B'
  // VCALL-NODIAG-NOT: runtime error: control flow integrity check {{.*}} during virtual call
  if (argv[1][2] == 'v') {
    a->f(); // UB here
  }

  // ICALL-FATAL-NOT: =3=
  // CAST-FATAL-NOT:  =3=
  // VCALL-FATAL-NOT: =3=
  // ALL-RECOVER: =3=
  fprintf(stderr, "=3=\n");

}
#endif
