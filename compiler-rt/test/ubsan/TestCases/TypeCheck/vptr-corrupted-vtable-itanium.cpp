// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr,null -g %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-CORRUPTED-VTABLE --strict-whitespace

// UNSUPPORTED: win32
// REQUIRES: stable-runtime, cxxabi
#include <cstddef>

#include <typeinfo>

struct S {
  S() {}
  ~S() {}
  virtual int v() { return 0; }
};

// See the proper definition in ubsan_type_hash_itanium.cc
struct VtablePrefix {
  signed long Offset;
  std::type_info *TypeInfo;
};

int main(int argc, char **argv) {
  // Test that we don't crash on corrupted vtable when
  // offset is too large or too small.
  S Obj;
  void *Ptr = &Obj;
  VtablePrefix* RealPrefix = reinterpret_cast<VtablePrefix*>(
      *reinterpret_cast<void**>(Ptr)) - 1;

  VtablePrefix Prefix[2];
  Prefix[0].Offset = 1<<21; // Greater than VptrMaxOffset
  Prefix[0].TypeInfo = RealPrefix->TypeInfo;

  // Hack Vtable ptr for Obj.
  *reinterpret_cast<void**>(Ptr) = static_cast<void*>(&Prefix[1]);

  // CHECK-CORRUPTED-VTABLE: vptr-corrupted-vtable-itanium.cpp:[[@LINE+3]]:16: runtime error: member call on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'S'
  // CHECK-CORRUPTED-VTABLE-NEXT: [[PTR]]: note: object has a possibly invalid vptr: abs(offset to top) too big
  S* Ptr2 = reinterpret_cast<S*>(Ptr);
  return Ptr2->v();
}
