// Test that we don't crash for vtable pointers with an invalid ptrauth
// signature which includes unauthenticated vtable pointers.

// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr -g %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// TODO(yln): introduce 'ptrauth' lit feature
// REQUIRES: stable-runtime, cxxabi, target={{arm64e.*}}

#include <typeinfo>
#include <ptrauth.h>

struct S {
  S() {}
  ~S() {}
  virtual int v() { return 0; }
};

int main(int argc, char **argv) {
  S Obj;
  void *Ptr = &Obj;
  void **VtablePtrPtr = reinterpret_cast<void **>(&Obj);
  // Hack Obj: the unauthenticated Vtable ptr will trigger an auth failure in the runtime.
  void *UnauthenticatedVtablePtr = ptrauth_strip(*VtablePtrPtr, 0);
  *VtablePtrPtr = UnauthenticatedVtablePtr;

  // CHECK: vptr-ptrauth-unauthenticated.cpp:[[@LINE+3]]:16: runtime error: member call on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'S'
  // CHECK: [[PTR]]: note: object has invalid vptr
  S *Ptr2 = reinterpret_cast<S *>(Ptr);
  return Ptr2->v();
}
