// RUN: %clangxx -g -DSHARED_LIB %s -fPIC -shared -o %T/target_uninstrumented-so.so
// RUN: %clangxx_cfi_diag -g %s -o %t %T/target_uninstrumented-so.so
// RUN: %t 2>&1 | FileCheck %s

// REQUIRES: cxxabi
// UNSUPPORTED: win32

#include <stdio.h>
#include <string.h>

struct A {
  virtual void f();
};

void *create_B();

#ifdef SHARED_LIB

struct B {
  virtual void f();
};
void B::f() {}

void *create_B() {
  return (void *)(new B());
}

#else

void A::f() {}

int main(int argc, char *argv[]) {
  void *p = create_B();
  // CHECK: runtime error: control flow integrity check for type 'A' failed during cast to unrelated type
  // CHECK: invalid vtable in module {{.*}}target_uninstrumented-so.so
  A *a = (A *)p;
  memset(p, 0, sizeof(A));
  // CHECK: runtime error: control flow integrity check for type 'A' failed during cast to unrelated type
  // CHECK-NOT: invalid vtable in module
  // CHECK: invalid vtable
  a = (A *)p;
  // CHECK: done
  fprintf(stderr, "done %p\n", a);
}
#endif
