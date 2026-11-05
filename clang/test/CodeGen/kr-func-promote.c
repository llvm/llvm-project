// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s %}

// CHECK: i32 @a(i32
int a();
int a(x) short x; {return x;}

// CHECK: void @b(double
// CHECK: %[[ADDR:.*]] = alloca float, align 4
// CHECK: %[[TRUNC:.*]] = fptrunc double %0 to float
// CHECK: store float %[[TRUNC]], ptr %[[ADDR]], align 4
void b();
void b(f) float f; {}
