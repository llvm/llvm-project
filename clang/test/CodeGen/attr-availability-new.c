// RUN: %clang_cc1 -fvisibility=hidden "-triple" "x86_64-apple-macos11.0" -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fvisibility=hidden "-triple" "x86_64-apple-macos10.15" -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-OLD %s

__attribute__((availability(macos,introduced=10.16)))
void f0(void);

__attribute__((availability(macos,introduced=11.0)))
void f1(void);

__attribute__((availability(macos,introduced=12.0)))
void f2(void);

// CHECK-OLD: declare extern_weak void @f0
// CHECK-OLD: declare extern_weak void @f1
// CHECK-OLD: declare extern_weak void @f2

// CHECK: declare void @f0
// CHECK: declare void @f1
// CHECK: declare extern_weak void @f2

void test() {
  f0();
  f1();
  f2();
}
