// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-extensions -emit-llvm < %s | FileCheck %s
// Test that __unaligned does not impact the layout of the fields.

struct A
{
    char a;
    __unaligned int b;
} a;
// CHECK: %struct.A = type { i8, i32 }

struct A2
{
    int b;
    char a;
    __unaligned int c;
} a2;
// CHECK: %struct.A2 = type { i32, i8, i32 }
