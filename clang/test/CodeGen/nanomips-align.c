// RUN: %clang_cc1 -w -triple nanomips-unknown-elf -emit-llvm -o - %s | FileCheck %s

// Global variables of 4 bytes and above should have at least 4-byte aligment
// unless specified otherwise via attributes.
char v1;
// CHECK: @v1 {{.*}}, align 1
short v2;
// CHECK: @v2 {{.*}}, align 2
int v3;
// CHECK: @v3 {{.*}}, align 4
int __attribute__((aligned(1))) v4;
// CHECK: @v4 {{.*}}, align 1
long long v5;
// CHECK: @v5 {{.*}}, align 8
long long __attribute__((aligned(16))) v6;
// CHECK: @v6 {{.*}}, align 16
char v7[2];
// CHECK: @v7 {{.*}}, align 1
char v8[5];
// CHECK: @v8 {{.*}}, align 4
char v9[4];
// CHECK: @v9 {{.*}}, align 4

