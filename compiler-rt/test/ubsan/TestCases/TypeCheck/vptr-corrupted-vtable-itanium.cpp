// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr -g %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-CORRUPTED-VTABLE --strict-whitespace

// UNSUPPORTED: windows-msvc
// REQUIRES: stable-runtime, cxxabi

#include <typeinfo>

#if __has_feature(ptrauth_calls)
#include <ptrauth.h>
#endif

struct S {
  S() {}
  ~S() {}
  virtual int v() { return 0; }
};

// See the proper definition in ubsan_type_hash_itanium.cpp
struct VtablePrefix {
  signed long Offset;
  std::type_info *TypeInfo;
};

int main(int argc, char **argv) {
  // Test that we don't crash on corrupted vtable when
  // offset is too large or too small.
  S Obj;
  void *Ptr = &Obj;
  void *VtablePtr = *reinterpret_cast<void**>(Ptr);
#if __has_feature(ptrauth_calls)
  VtablePtr = ptrauth_strip(VtablePtr, 0);
#endif
  VtablePrefix* Prefix = reinterpret_cast<VtablePrefix*>(VtablePtr) - 1;

  VtablePrefix FakePrefix[2];
  FakePrefix[0].Offset = 1<<21; // Greater than VptrMaxOffset
  FakePrefix[0].TypeInfo = Prefix->TypeInfo;

  // Hack Vtable ptr for Obj.
  void *FakeVtablePtr = static_cast<void*>(&FakePrefix[1]);
#if __has_feature(ptrauth_calls)
  FakeVtablePtr = ptrauth_sign_unauthenticated(
      FakeVtablePtr, ptrauth_key_cxx_vtable_pointer, 0);
#endif
  *reinterpret_cast<void**>(Ptr) = FakeVtablePtr;

  // CHECK-CORRUPTED-VTABLE: vptr-corrupted-vtable-itanium.cpp:[[@LINE+3]]:16: runtime error: member call on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'S'
  // CHECK-CORRUPTED-VTABLE-NEXT: [[PTR]]: note: object has a possibly invalid vptr: abs(offset to top) too big
  S* Ptr2 = reinterpret_cast<S*>(Ptr);
  return Ptr2->v();
}
