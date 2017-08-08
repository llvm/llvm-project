// RUN: %clangxx_cfi_dso -DSHARED_LIB %s -fPIC -shared -o %t1-so.so
// RUN: %clangxx_cfi_dso %s -o %t1 %t1-so.so
// RUN: %t1 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_dso -DB32 -DSHARED_LIB %s -fPIC -shared -o %t2-so.so
// RUN: %clangxx_cfi_dso -DB32 %s -o %t2 %t2-so.so
// RUN: %t2 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_dso -DB64 -DSHARED_LIB %s -fPIC -shared -o %t3-so.so
// RUN: %clangxx_cfi_dso -DB64 %s -o %t3 %t3-so.so
// RUN: %t3 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_dso -DBM -DSHARED_LIB %s -fPIC -shared -o %t4-so.so
// RUN: %clangxx_cfi_dso -DBM %s -o %t4 %t4-so.so
// RUN: %t4 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -DBM -DSHARED_LIB %s -fPIC -shared -o %t5-so.so
// RUN: %clangxx -DBM %s -o %t5 %t5-so.so
// RUN: %t5 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t5 x 2>&1 | FileCheck --check-prefix=NCFI %s

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
