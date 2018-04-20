// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NONE
// RUN: %clang_cc1 -emit-llvm -function-alignment 4 %s -o - | FileCheck %s -check-prefix CHECK-16
// RUN: %clang_cc1 -emit-llvm -function-alignment 5 %s -o - | FileCheck %s -check-prefix CHECK-32

void f(void) {}
void __attribute__((__aligned__(64))) g(void) {}

// CHECK-NONE-NOT: define {{(dso_local )?}}void @f() #0 align
// CHECK-NONE: define {{(dso_local )?}}void @g() #0 align 64

// CHECK-16: define {{(dso_local )?}}void @f() #0 align 16
// CHECK-16: define {{(dso_local )?}}void @g() #0 align 64

// CHECK-32: define {{(dso_local )?}}void @f() #0 align 32
// CHECK-32: define {{(dso_local )?}}void @g() #0 align 64

