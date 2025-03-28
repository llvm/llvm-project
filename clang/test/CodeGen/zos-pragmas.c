// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -emit-llvm -triple s390x-none-zos -fvisibility=hidden %s -o - | FileCheck %s

int a,b,c;
#pragma export(a) export(b) export(c)

void foo(void);

// CHECK: @a = global i32 0, align 4
// CHECK: @b = global i32 0, align 4
// CHECK: @c = global i32 0, align 4
