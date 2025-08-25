// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: %clangxx_cfi_dso -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso -g %s -o %t %ld_flags_rpath_exe
// RUN: %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_dso -DB32 -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso -DB32 %s -o %t %ld_flags_rpath_exe
// RUN: %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_dso -DB64 -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso -DB64 %s -o %t %ld_flags_rpath_exe
// RUN: %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_dso -DBM -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_dso -DBM %s -o %t %ld_flags_rpath_exe
// RUN: %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -DBM -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx -DBM %s -o %t %ld_flags_rpath_exe
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t x 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism crashes the program when making a virtual call
// to an object of the wrong class but with a compatible vtable, by casting a
// pointer to such an object and attempting to make a call through it.

// REQUIRES: cxxabi

#include <stdio.h>
#include <string.h>

struct A {
  virtual void f();
};

A *create_B();

#ifdef SHARED_LIB

#include "../utils.h"
struct B : public A {
  virtual void f();
};
void B::f() {}

A *create_B() {
  create_derivers<B>();
  return new B();
}

#else

void A::f() {}

int main(int argc, char *argv[]) {
  A *a = create_B();

  // CFI: =1=
  // NCFI: =1=
  fprintf(stderr, "=1=\n");
  a->f(); // OK
  // CFI: =2=
  // NCFI: =2=
  fprintf(stderr, "=2=\n");
}
#endif
