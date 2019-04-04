// RUN: %clang_cc1 %s -triple spir -cl-std=c++ -emit-llvm -O0 -o - | FileCheck %s

// Test that we don't initialize local address space objects.
//CHECK: @_ZZ4testvE1i = internal addrspace(3) global i32 undef
//CHECK: @_ZZ4testvE2ii = internal addrspace(3) global %class.C undef
class C {
  int i;
};

kernel void test() {
  __local int i;
  __local C ii;
  // FIXME: In OpenCL C we don't accept initializers for local
  // address space variables. User defined initialization could
  // make sense, but would it mean that all work items need to
  // execute it? Potentially disallowing any initialization would
  // make things easier and assingments can be used to set specific
  // values. This rules should make it consistent with OpenCL C.
  //__local C c();
}
