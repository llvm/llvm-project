// Global variables of intergal types
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o -  | FileCheck %s

// Note: Currently unsupported features include default zero-initialization
//       and alignment. The fact that "external" is only printed for globals
//       without an initializer is a quirk of the LLVM AsmWriter.

char c;
// CHECK: @c = external dso_local global i8

signed char sc;
// CHECK: @sc = external dso_local global i8

unsigned char uc;
// CHECK: @uc = external dso_local global i8

short ss;
// CHECK: @ss = external dso_local global i16

unsigned short us = 100;
// CHECK: @us = dso_local global i16 100

int si = 42;
// CHECK: @si = dso_local global i32 42

unsigned ui;
// CHECK: @ui = external dso_local global i32

long sl;
// CHECK: @sl = external dso_local global i64

unsigned long ul;
// CHECK: @ul = external dso_local global i64

long long sll;
// CHECK: @sll = external dso_local global i64

unsigned long long ull = 123456;
// CHECK: @ull = dso_local global i64 123456

__int128 s128;
// CHECK: @s128 = external dso_local global i128

unsigned __int128 u128;
// CHECK: @u128 = external dso_local global i128

wchar_t wc;
// CHECK: @wc = external dso_local global i32

char8_t c8;
// CHECK: @c8 = external dso_local global i8

char16_t c16;
// CHECK: @c16 = external dso_local global i16

char32_t c32;
// CHECK: @c32 = external dso_local global i32

_BitInt(20) sb20;
// CHECK: @sb20 = external dso_local global i20

unsigned _BitInt(48) ub48;
// CHECK: @ub48 = external dso_local global i48

_Float16 f16;
// CHECK: @f16 = external dso_local global half

__bf16 bf16;
// CHECK: @bf16 = external dso_local global bfloat

float f;
// CHECK: @f = external dso_local global float

double d = 1.25;
// CHECK: @d = dso_local global double 1.250000e+00

long double ld;
// CHECK: @ld = external dso_local global x86_fp80

__float128 f128;
// CHECK: @f128 = external dso_local global fp128
