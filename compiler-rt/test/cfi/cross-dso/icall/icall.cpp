// RUN: %clangxx_cfi_dso -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso %s -o %t %ld_flags_rpath_exe && %expect_crash %t 2>&1 | FileCheck %s

// RUN: %clangxx_cfi_dso_diag -g -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso_diag -g %s -o %t %ld_flags_rpath_exe && %t 2>&1 | FileCheck %s --check-prefix=CFI-DIAG

#include <stdio.h>

#ifdef SHARED_LIB
void f() {
}
#else
void f();
int main() {
  // CHECK-DIAG: =1=
  // CHECK: =1=
  fprintf(stderr, "=1=\n");
  ((void (*)(void))f)();
  // CHECK-DIAG: =2=
  // CHECK: =2=
  fprintf(stderr, "=2=\n");
  // CFI-DIAG: runtime error: control flow integrity check for type 'void (int)' failed during indirect function call
  // CFI-DIAG-NEXT: note: f() defined here
  ((void (*)(int))f)(42); // UB here
  // CHECK-DIAG: =3=
  // CHECK-NOT: =3=
  fprintf(stderr, "=3=\n");
}
#endif
