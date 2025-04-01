// Global variables of intergal types
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s

// Note: Currently unsupported features include alignment..

char c;
// CHECK: @c = dso_local global i8 0

signed char sc;
// CHECK: @sc = dso_local global i8 0

unsigned char uc;
// CHECK: @uc = dso_local global i8 0

short ss;
// CHECK: @ss = dso_local global i16 0

unsigned short us = 100;
// CHECK: @us = dso_local global i16 100

int si = 42;
// CHECK: @si = dso_local global i32 42

unsigned ui;
// CHECK: @ui = dso_local global i32 0

long sl;
// CHECK: @sl = dso_local global i64 0

unsigned long ul;
// CHECK: @ul = dso_local global i64 0

long long sll;
// CHECK: @sll = dso_local global i64 0

unsigned long long ull = 123456;
// CHECK: @ull = dso_local global i64 123456

__int128 s128;
// CHECK: @s128 = dso_local global i128 0

unsigned __int128 u128;
// CHECK: @u128 = dso_local global i128 0

wchar_t wc;
// CHECK: @wc = dso_local global i32 0

char8_t c8;
// CHECK: @c8 = dso_local global i8 0

char16_t c16;
// CHECK: @c16 = dso_local global i16 0

char32_t c32;
// CHECK: @c32 = dso_local global i32 0

_BitInt(20) sb20;
// CHECK: @sb20 = dso_local global i20 0

unsigned _BitInt(48) ub48;
// CHECK: @ub48 = dso_local global i48 0

bool boolfalse = false;
// CHECK: @boolfalse = dso_local global i8 0

_Float16 f16;
// CHECK: @f16 = dso_local global half

__bf16 bf16;
// CHECK: @bf16 = dso_local global bfloat

float f;
// CHECK: @f = dso_local global float 0.000000e+00

double d = 1.25;
// CHECK: @d = dso_local global double 1.250000e+00

long double ld;
// CHECK: @ld = dso_local global x86_fp80 0xK00

__float128 f128;
// CHECK: @f128 = dso_local global fp128 0xL00

void *vp;
// CHECK: @vp = dso_local global ptr null

int *ip = 0;
// CHECK: @ip = dso_local global ptr null

double *dp;
// CHECK: @dp = dso_local global ptr null

char **cpp;
// CHECK: @cpp = dso_local global ptr null

void (*fp)();
// CHECK: @fp = dso_local global ptr null

int (*fpii)(int) = 0;
// CHECK: @fpii = dso_local global ptr null

void (*fpvar)(int, ...);
// CHECK: @fpvar = dso_local global ptr null
