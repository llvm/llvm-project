/// This test verifies IR generated for APIs protected with availability annotations with a common versions.
// RUN: %clang_cc1 -fvisibility=hidden "-triple" "arm64-apple-ios26.0" -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fvisibility=hidden "-triple" "arm64-apple-tvos26" -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fvisibility=hidden "-triple" "arm64-apple-watchos26" -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fvisibility=hidden "-triple" "arm64-apple-ios18" -emit-llvm -o - %s | FileCheck -check-prefix=OLD %s

__attribute__((availability(ios,introduced=19)))
void f0(void);

__attribute__((availability(ios,introduced=26)))
void f1(void);

__attribute__((availability(ios,introduced=27)))
void f2(void);

// OLD: declare extern_weak void @f0
// OLD: declare extern_weak void @f1
// OLD: declare extern_weak void @f2

// CHECK:  declare void @f0
// CHECK:  declare void @f1
// CHECK:  declare extern_weak void @f2

void test() {
  f0();
  f1();
  f2();
}
