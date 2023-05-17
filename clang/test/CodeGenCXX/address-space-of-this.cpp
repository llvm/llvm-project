// RUN: %clang_cc1 %s -std=c++14 -triple=spir -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -std=c++17 -triple=spir -emit-llvm -o - | FileCheck %s
// XFAIL: *

// FIXME: We can't compile address space method qualifiers yet.
// Therefore there is no way to check correctness of this code.
struct MyType {
  MyType(int i) __attribute__((address_space(10))) : i(i) {}
  int i;
};
//CHECK: call void @_ZN6MyTypeC1Ei(ptr addrspacecast (ptr addrspace(10) @m to ptr), i32 123)
MyType __attribute__((address_space(10))) m = 123;
