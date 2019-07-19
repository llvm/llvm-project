// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -emit-llvm -O0 -o - | FileCheck %s

struct MyType {
  MyType(int i) : i(i) {}
  MyType(int i) __constant : i(i) {}
  int i;
};

//CHECK: call void @_ZNU3AS26MyTypeC1Ei(%struct.MyType addrspace(2)* @const1, i32 1)
__constant MyType const1 = 1;
//CHECK: call void @_ZNU3AS26MyTypeC1Ei(%struct.MyType addrspace(2)* @const2, i32 2)
__constant MyType const2(2);
//CHECK: call void @_ZNU3AS46MyTypeC1Ei(%struct.MyType addrspace(4)* addrspacecast (%struct.MyType addrspace(1)* @glob to %struct.MyType addrspace(4)*), i32 1)
MyType glob(1);
