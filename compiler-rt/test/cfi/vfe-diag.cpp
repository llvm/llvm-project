// RUN: %clangxx_cfi_diag -fno-sanitize-recover=cfi -fvirtual-function-elimination -o %t %s
// RUN: not %run %t 2>&1 | FileCheck %s

// Tests that diagnostic CFI reports a bad virtual call with
// -fvirtual-function-elimination, which emits llvm.type.checked.load for
// every virtual call.

// -fvirtual-function-elimination requires full LTO.
// UNSUPPORTED: thinlto

#include "utils.h"
#include <string.h>

struct A {
  virtual void f();
};
void A::f() {}

// A second implementation keeps WPD from devirtualizing a->f().
struct A2 : A {
  void f() override;
};
void A2::f() {}

struct B {
  virtual void f();
};
void B::f() {}

int main() {
  break_optimization(new A2);
  B *b = new B;
  break_optimization(b);
  A *a;
  memcpy(&a, &b, sizeof(a)); // Avoid the cast checks.
  // CHECK: runtime error: control flow integrity check for type 'A' failed during virtual call
  // CHECK-NEXT: note: vtable is of type '{{(struct )?}}B'
  a->f();
}
