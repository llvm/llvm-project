// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-out-of-line-declaration -emit-llvm -o - %s | FileCheck %s

void foo() {
  struct S {
    void bar();
  };
  void S::bar();
}

// CHECK-LABEL: define {{.*}}void @_Z3foov()
// CHECK: ret void
