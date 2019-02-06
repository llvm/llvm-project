// RUN: %clang_cc1 -triple msp430-elf -emit-llvm %s -o - | FileCheck %s

// MSP430 target prefers chars to be aligned to 8 bit and other types to 16 bit.

// CHECK: @c ={{.*}}global i8 1, align 1
// CHECK: @s ={{.*}}global i16 266, align 2
// CHECK: @i ={{.*}}global i16 266, align 2
// CHECK: @l ={{.*}}global i32 16909060, align 2
// CHECK: @ll ={{.*}}global i64 283686952306183, align 2
// CHECK: @f ={{.*}}global float 1.000000e+00, align 2
// CHECK: @d ={{.*}}global double 1.000000e+00, align 2
// CHECK: @ld ={{.*}}global double 1.000000e+00, align 2
// CHECK: @p ={{.*}}global i8* @c, align 2

char c = 1;
short s = 266;
int i = 266;
long l = 16909060;
long long ll = 283686952306183;
float f = 1.0f;
double d = 1.0;
long double ld = 1.0;
char *p = &c;
