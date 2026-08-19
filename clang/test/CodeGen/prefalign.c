// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux -fpreferred-function-alignment=16 %s -o - | FileCheck %s

// CHECK: define {{.*}} void @f() {{.*}} prefalign(16)
void f() {}
