// RUN: %clang_cc1 %s -triple spir -cl-std=clc++ -emit-llvm -O0 -o - | FileCheck %s

struct B {
  int mb;
};

class D : public B {
public:
  int getmb() { return mb; }
};

void foo() {
  D d;
  //CHECK: addrspacecast %class.D* %d to %class.D addrspace(4)*
  //CHECK: call spir_func i32 @_ZNU3AS41D5getmbEv(%class.D addrspace(4)*
  d.getmb();
}

//Derived and Base are in the same address space.

//CHECK: define linkonce_odr spir_func i32 @_ZNU3AS41D5getmbEv(%class.D addrspace(4)* %this)
//CHECK: bitcast %class.D addrspace(4)* %this1 to %struct.B addrspace(4)*
