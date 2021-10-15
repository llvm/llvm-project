// RUN: %clang_cc1 -fno-experimental-new-pass-manager -emit-llvm -o - -triple x86_64-apple-darwin10 %s | FileCheck %s
// RUN: %clang_cc1 -fno-experimental-new-pass-manager -O2 -fno-inline -emit-llvm -o - -triple x86_64-apple-darwin10 %s | FileCheck %s
// RUN: %clang_cc1 -fno-experimental-new-pass-manager -flto -O2 -fno-inline -emit-llvm -o - -triple x86_64-apple-darwin10 %s | FileCheck %s -check-prefix=LTO
// RUN: %clang_cc1 -fexperimental-new-pass-manager -emit-llvm -o - -triple x86_64-apple-darwin10 %s | FileCheck %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -O2 -fno-inline -emit-llvm -o - -triple x86_64-apple-darwin10 %s | FileCheck %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flto -O2 -fno-inline -emit-llvm -o - -triple x86_64-apple-darwin10 %s | FileCheck %s -check-prefix=LTO

// Ensure that we don't emit available_externally functions at -O0.
// Also should not emit them at -O2, unless -flto is present in which case
// we should preserve them for link-time inlining decisions.
int x;

inline void f0(int y) { x = y; }

// CHECK-LABEL: define{{.*}} void @test()
// CHECK: declare void @f0(i32 noundef)
// LTO-LABEL: define{{.*}} void @test()
// LTO: define available_externally void @f0
void test() {
  f0(17);
}

inline int __attribute__((always_inline)) f1(int x) {
  int blarg = 0;
  for (int i = 0; i < x; ++i)
    blarg = blarg + x * i;
  return blarg;
}

// CHECK: @test1
// LTO: @test1
int test1(int x) {
  // CHECK-NOT: call {{.*}} @f1
  // CHECK: ret i32
  // LTO-NOT: call {{.*}} @f1
  // LTO: ret i32
  return f1(x);
}
