// RUN: %clang_cc1 -triple spirv64-intel %s -emit-llvm -o - | FileCheck %s

// CHECK: all spir_func addrspace(9) i32 @__cxa_atexit(ptr addrspace(4) addrspacecast (ptr addrspace(9) @{{.*}} to ptr addrspace(4)),
struct S {
  ~S() {}
};
S s;
