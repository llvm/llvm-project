// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=16 %s -o - | FileCheck %s --check-prefix=CHECK-16
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=32 %s -o - | FileCheck %s --check-prefix=CHECK-32

// CHECK-16: define {{.*}} void @f() {{.*}} prefalign(16)
// CHECK-32: define {{.*}} void @f() {{.*}} prefalign(32)

void f() {}
