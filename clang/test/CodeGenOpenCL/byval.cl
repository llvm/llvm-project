// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn---opencl %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn---amdgizcl %s | FileCheck %s -check-prefix=AMDGIZ

struct A {
  int x[100];
};

int f(struct A a);

int g() {
  struct A a;
  // CHECK: call i32 @f(%struct.A* byval{{.*}}%a)
  // AMDGIZ: call i32 @f(%struct.A addrspace(5)* byval{{.*}}%a)
  return f(a);
}

// CHECK: declare i32 @f(%struct.A* byval{{.*}})
// AMDGIZ: declare i32 @f(%struct.A addrspace(5)* byval{{.*}})
