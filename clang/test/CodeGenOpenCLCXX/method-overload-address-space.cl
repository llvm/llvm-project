//RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -emit-llvm -O0 -o - | FileCheck %s

struct C {
  void foo() __local;
  void foo() __global;
  void foo();
  void bar();
};

__global C c1;

__kernel void k() {
  __local C c2;
  C c3;
  __global C &c_ref = c1;
  __global C *c_ptr;

  // CHECK: call void @_ZNU3AS11C3fooEv(%struct.C addrspace(1)*
  c1.foo();
  // CHECK: call void @_ZNU3AS31C3fooEv(%struct.C addrspace(3)*
  c2.foo();
  // CHECK: call void @_ZNU3AS41C3fooEv(%struct.C addrspace(4)*
  c3.foo();
  // CHECK: call void @_ZNU3AS11C3fooEv(%struct.C addrspace(1)*
  c_ptr->foo();
  // CHECK: void @_ZNU3AS11C3fooEv(%struct.C addrspace(1)*
  c_ref.foo();

  // CHECK: call void @_ZNU3AS41C3barEv(%struct.C addrspace(4)* addrspacecast (%struct.C addrspace(1)* @c1 to %struct.C addrspace(4)*))
  c1.bar();
  //FIXME: Doesn't compile yet
  //c_ptr->bar();
  // CHECK: call void @_ZNU3AS41C3barEv(%struct.C addrspace(4)* addrspacecast (%struct.C addrspace(1)* @c1 to %struct.C addrspace(4)*))
  c_ref.bar();
}
