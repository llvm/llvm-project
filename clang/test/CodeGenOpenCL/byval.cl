// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn %s | FileCheck %s

struct A {
  int x[100];
};

int f(struct A a);

int g() {
  struct A a;
  // CHECK: call i32 @f(%struct.A addrspace(5)* byval{{.*}}%a)
  return f(a);
}

// CHECK: declare i32 @f(%struct.A addrspace(5)* byval{{.*}})
