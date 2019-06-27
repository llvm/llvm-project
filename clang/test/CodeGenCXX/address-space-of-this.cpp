// RUN: %clang_cc1 %s -std=c++14 -triple=spir -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -std=c++17 -triple=spir -emit-llvm -o - | FileCheck %s

struct MyType {
  MyType(int i) : i(i) {}
  int i;
};
//CHECK: call void @_ZN6MyTypeC1Ei(%struct.MyType* addrspacecast (%struct.MyType addrspace(10)* @m to %struct.MyType*), i32 123)
MyType __attribute__((address_space(10))) m = 123;
