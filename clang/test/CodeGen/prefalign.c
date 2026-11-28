// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux %s -o - | FileCheck %s -check-prefix CHECK-NONE
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=1 %s -o - | FileCheck %s -check-prefix CHECK-1
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=16 %s -o - | FileCheck %s -check-prefix CHECK-16
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=32 %s -o - | FileCheck %s -check-prefix CHECK-32
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=64 %s -o - | FileCheck %s -check-prefix CHECK-64
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=65536 %s -o - | FileCheck %s -check-prefix CHECK-65536
// RUN: not %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=3 %s -o - 2>&1 | FileCheck %s -check-prefix CHECK-ERR-3
// RUN: not %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=131072 %s -o - 2>&1 | FileCheck %s -check-prefix CHECK-ERR-131072

// CHECK-NONE-NOT: prefalign(
// CHECK-1: define {{.*}} void @f() {{.*}} prefalign(1)
// CHECK-16: define {{.*}} void @f() {{.*}} prefalign(16)
// CHECK-32: define {{.*}} void @f() {{.*}} prefalign(32)
// CHECK-64: define {{.*}} void @f() {{.*}} prefalign(64)
// CHECK-65536: define {{.*}} void @f() {{.*}} prefalign(65536)
// CHECK-ERR-3: alignment is not a power of 2 in '-fpreferred-function-alignment=3'
// CHECK-ERR-131072: invalid integral value '131072' in '-fpreferred-function-alignment=131072'
void f() {}
