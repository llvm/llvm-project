// RUN: %clangxx_cfi_dso_diag -std=c++11 %s -o %t
// RUN: %t zero 2>&1 | FileCheck --check-prefix=CHECK-ZERO %s
// RUN: %t unaddressable 2>&1 | FileCheck --check-prefix=CHECK-UNADDR %s
// RUN: %t 2>&1 | FileCheck --check-prefix=CHECK-TYPEINFO %s

// RUN: %clangxx_cfi_diag -std=c++11 %s -o %t2
// RUN: %t2 zero 2>&1 | FileCheck --check-prefix=CHECK-ZERO %s
// RUN: %t2 unaddressable 2>&1 | FileCheck --check-prefix=CHECK-UNADDR %s
// RUN: %t2 2>&1 | FileCheck --check-prefix=CHECK-TYPEINFO %s

// REQUIRES: cxxabi

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

struct A {
  virtual void f();
};

void A::f() {}

int main(int argc, char *argv[]) {
  char *volatile p = reinterpret_cast<char *>(new A());
  if (argc > 1 && strcmp(argv[1], "unaddressable") == 0) {
    void *vtable = mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    // Create an object with a vtable in an unaddressable memory region.
    *(uintptr_t *)p = (uintptr_t)vtable + 64;
    // CHECK-UNADDR: runtime error: control flow integrity check for type 'A' failed during cast
    // CHECK-UNADDR: note: invalid vtable
    // CHECK-UNADDR: <memory cannot be printed>
    // CHECK-UNADDR: runtime error: control flow integrity check for type 'A' failed during cast
    // CHECK-UNADDR: note: invalid vtable
    // CHECK-UNADDR: <memory cannot be printed>
  } else if (argc > 1 && strcmp(argv[1], "zero") == 0) {
    // Create an object with a vtable outside of any known DSO, but still in an
    // addressable area.
    void *vtable = calloc(1, 128);
    *(uintptr_t *)p = (uintptr_t)vtable + 64;
    // CHECK-ZERO: runtime error: control flow integrity check for type 'A' failed during cast
    // CHECK-ZERO: note: invalid vtable
    // CHECK-ZERO: 00 00 00 00 00 00 00 00
    // CHECK-ZERO: runtime error: control flow integrity check for type 'A' failed during cast
    // CHECK-ZERO: note: invalid vtable
    // CHECK-ZERO: 00 00 00 00 00 00 00 00
  } else {
    // Create an object with a seemingly fine vtable, but with an unaddressable
    // typeinfo pointer.
    void *vtable = calloc(1, 128);
    memset(vtable, 0xFE, 128);
    *(uintptr_t *)p = (uintptr_t)vtable + 64;
    // CHECK-TYPEINFO: runtime error: control flow integrity check for type 'A' failed during cast
    // CHECK-TYPEINFO: note: invalid vtable
    // CHECK-TYPEINFO: fe fe fe fe fe fe fe fe
    // CHECK-TYPEINFO: runtime error: control flow integrity check for type 'A' failed during cast
    // CHECK-TYPEINFO: note: invalid vtable
    // CHECK-TYPEINFO: fe fe fe fe fe fe fe fe
  }

  A *volatile pa = reinterpret_cast<A *>(p);
  pa = reinterpret_cast<A *>(p);
}
