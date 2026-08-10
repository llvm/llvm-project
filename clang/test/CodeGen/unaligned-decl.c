// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-extensions -emit-llvm < %s | FileCheck %s

// CHECK: @a1 ={{.*}} global i32 1, align 1
__unaligned int a1 = 1;

// CHECK: @a2 ={{.*}} global i32 1, align 1
int __unaligned a2 = 1;

// CHECK: @a3 = {{.*}} align 1
__unaligned int a3[10];

// CHECK: @a4 = {{.*}} align 1
int __unaligned a4[10];

// CHECK: @p1 = {{.*}} align 1
int *__unaligned p1;

// CHECK: @p2 = {{.*}} align 8
int __unaligned *p2;

// CHECK: @p3 = {{.*}} align 1
int __unaligned *__unaligned p3;
