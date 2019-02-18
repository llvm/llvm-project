// RUN: %clangxx -g -DSHARED_LIB %s -fPIC -shared -o %dynamiclib %ld_flags_rpath_so
// RUN: %clangxx_cfi_diag -g %s -o %t %ld_flags_rpath_exe
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: cxxabi
// UNSUPPORTED: windows-msvc

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
  // CHECK: invalid vtable
  // CHECK: check failed in {{.*}}, vtable located in {{.*}}libtarget_uninstrumented.cpp.dynamic.so
  A *a = (A *)p;
  memset(p, 0, sizeof(A));

  // CHECK: runtime error: control flow integrity check for type 'A' failed during cast to unrelated type
  // CHECK: invalid vtable
  // CHECK: check failed in {{.*}}, vtable located in (unknown)
  a = (A *)p;
  // CHECK: done
  fprintf(stderr, "done %p\n", a);
}
#endif
