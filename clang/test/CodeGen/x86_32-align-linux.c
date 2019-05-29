// RUN: %clang_cc1 -w -fblocks -ffreestanding -triple i386-pc-linux-gnu -emit-llvm -o %t %s
// RUN: FileCheck < %t %s

#include <immintrin.h>

typedef union {
        int d[4];
        __m128 m;
} M128;

extern void foo(int, ...);

M128 a;

// CHECK-LABEL: define void @test
// CHECK: entry:
// CHECK: call void (i32, ...) @foo(i32 1, %union.M128* byval align 16
// CHECK: call void (i32, ...) @foo(i32 1, <4 x float>

void test(void)
{
  foo(1, a);
  foo(1, a.m);
}

