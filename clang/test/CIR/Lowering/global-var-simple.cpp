// Global variables of intergal types
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --input-file %t-cir.ll
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=OGCG --input-file %t.ll

// Note: The differences between CHECK and OGCG represent incorrect IR lowering
//       with ClangIR enabled and will be fixed in a future patch.

char c;
// CHECK: @c = global i8 0, align 1
// OGCG:  @c = global i8 0, align 1

signed char sc;
// CHECK: @sc = global i8 0, align 1
// OGCG:  @sc = global i8 0, align 1

unsigned char uc;
// CHECK: @uc = global i8 0, align 1
// OGCG:  @uc = global i8 0, align 1

short ss;
// CHECK: @ss = global i16 0, align 2
// OGCG:  @ss = global i16 0, align 2

unsigned short us = 100;
// CHECK: @us = global i16 100, align 2
// OGCG:  @us = global i16 100, align 2

int si = 42;
// CHECK: @si = global i32 42, align 4
// OGCG:  @si = global i32 42, align 4

unsigned ui;
// CHECK: @ui = global i32 0, align 4
// OGCG:  @ui = global i32 0, align 4

long sl;
// CHECK: @sl = global i64 0, align 8
// OGCG:  @sl = global i64 0, align 8

unsigned long ul;
// CHECK: @ul = global i64 0, align 8
// OGCG:  @ul = global i64 0, align 8

long long sll;
// CHECK: @sll = global i64 0, align 8
// OGCG:  @sll = global i64 0, align 8

unsigned long long ull = 123456;
// CHECK: @ull = global i64 123456, align 8
// OGCG:  @ull = global i64 123456, align 8

__int128 s128;
// CHECK: @s128 = global i128 0, align 16
// OGCG:  @s128 = global i128 0, align 16

unsigned __int128 u128;
// CHECK: @u128 = global i128 0, align 16
// OGCG:  @u128 = global i128 0, align 16

wchar_t wc;
// CHECK: @wc = global i32 0, align 4
// OGCG:  @wc = global i32 0, align 4

char8_t c8;
// CHECK: @c8 = global i8 0, align 1
// OGCG:  @c8 = global i8 0, align 1

char16_t c16;
// CHECK: @c16 = global i16 0, align 2
// OGCG:  @c16 = global i16 0, align 2

char32_t c32;
// CHECK: @c32 = global i32 0, align 4
// OGCG:  @c32 = global i32 0, align 4

_BitInt(20) sb20;
// CHECK: @sb20 = global i20 0, align 4
// OGCG:  @sb20 = global i32 0, align 4

unsigned _BitInt(48) ub48;
// CHECK: @ub48 = global i48 0, align 8
// OGCG:  @ub48 = global i64 0, align 8

bool boolfalse = false;
// CHECK: @boolfalse = global i8 0, align 1
// OGCG:  @boolfalse = global i8 0, align 1

_Float16 f16;
// CHECK: @f16 = global half 0xH0000, align 2
// OGCG:  @f16 = global half 0xH0000, align 2

__bf16 bf16;
// CHECK: @bf16 = global bfloat 0xR0000, align 2
// OGCG:  @bf16 = global bfloat 0xR0000, align 2

float f;
// CHECK: @f = global float 0.000000e+00, align 4
// OGCG:  @f = global float 0.000000e+00, align 4

double d = 1.25;
// CHECK: @d = global double 1.250000e+00, align 8
// OGCG:  @d = global double 1.250000e+00, align 8

long double ld;
// CHECK: @ld = global x86_fp80 0xK00000000000000000000, align 16
// OGCG:  @ld = global x86_fp80 0xK00000000000000000000, align 16

__float128 f128;
// CHECK: @f128 = global fp128 0xL00000000000000000000000000000000, align 16
// OGCG:  @f128 = global fp128 0xL00000000000000000000000000000000, align 16

void *vp;
// CHECK: @vp = global ptr null, align 8
// OGCG:  @vp = global ptr null, align 8

int *ip = 0;
// CHECK: @ip = global ptr null, align 8
// OGCG:  @ip = global ptr null, align 8

double *dp;
// CHECK: @dp = global ptr null, align 8
// OGCG:  @dp = global ptr null, align 8

char **cpp;
// CHECK: @cpp = global ptr null, align 8
// OGCG:  @cpp = global ptr null, align 8

void (*fp)();
// CHECK: @fp = global ptr null, align 8
// OGCG:  @fp = global ptr null, align 8

int (*fpii)(int) = 0;
// CHECK: @fpii = global ptr null, align 8
// OGCG:  @fpii = global ptr null, align 8

void (*fpvar)(int, ...);
// CHECK: @fpvar = global ptr null, align 8
// OGCG:  @fpvar = global ptr null, align 8
