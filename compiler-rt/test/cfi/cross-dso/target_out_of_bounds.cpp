// RUN: %clangxx_cfi_dso_diag %s -o %t
// RUN: %expect_crash %t 2>&1 | FileCheck %s

// REQUIRES: cxxabi

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct A {
  virtual void f();
};

void A::f() {}

int main(int argc, char *argv[]) {
  // Create an object with a vtable outside of any known DSO, but still in an
  // addressable area. Current implementation of handlers in UBSan is not robust
  // enough to handle unaddressable vtables. TODO: fix this.
  void *empty = calloc(1, 128);
  uintptr_t v = (uintptr_t)empty + 64;
  char *volatile p = reinterpret_cast<char *>(new A());
  for (uintptr_t *q = (uintptr_t *)p; q < (uintptr_t *)(p + sizeof(A)); ++q)
    *q = v;

  // CHECK: runtime error: control flow integrity check for type 'A' failed during cast
  A *volatile pa = reinterpret_cast<A *>(p);

  // CHECK: untime error: control flow integrity check for type 'A' failed during virtual call
  pa->f();
}
