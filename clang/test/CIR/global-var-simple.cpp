// Global variables of scalar typees with initial values
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s

char c;
// CHECK: cir.global external @c : !s8i

signed char sc;
// CHECK: cir.global external @sc : !s8i

unsigned char uc;
// CHECK: cir.global external @uc : !u8i

short ss;
// CHECK: cir.global external @ss : !s16i

unsigned short us = 100;
// CHECK: cir.global external @us = #cir.int<100> : !u16i

int si = 42;
// CHECK: cir.global external @si = #cir.int<42> : !s32i

unsigned ui;
// CHECK: cir.global external @ui : !u32i

long sl;
// CHECK: cir.global external @sl : !s64i

unsigned long ul;
// CHECK: cir.global external @ul : !u64i

long long sll;
// CHECK: cir.global external @sll : !s64i

unsigned long long ull = 123456;
// CHECK: cir.global external @ull = #cir.int<123456> : !u64i

__int128 s128;
// CHECK: cir.global external @s128 : !s128i

unsigned __int128 u128;
// CHECK: cir.global external @u128 : !u128i

wchar_t wc;
// CHECK: cir.global external @wc : !s32i

char8_t c8;
// CHECK: cir.global external @c8 : !u8i

char16_t c16;
// CHECK: cir.global external @c16 : !u16i

char32_t c32;
// CHECK: cir.global external @c32 : !u32i

_BitInt(20) sb20;
// CHECK: cir.global external @sb20 : !cir.int<s, 20>

unsigned _BitInt(48) ub48;
// CHECK: cir.global external @ub48 : !cir.int<u, 48>

bool boolfalse = false;
// CHECK: cir.global external @boolfalse = #false

_Float16 f16;
// CHECK: cir.global external @f16 : !cir.f16

__bf16 bf16;
// CHECK: cir.global external @bf16 : !cir.bf16

float f;
// CHECK: cir.global external @f : !cir.float

double d = 1.25;
// CHECK: cir.global external @d = #cir.fp<1.250000e+00> : !cir.double

long double ld;
// CHECK: cir.global external @ld : !cir.long_double<!cir.f80>

__float128 f128;
// CHECK: cir.global external @f128 : !cir.f128

void *vp;
// CHECK: cir.global external @vp : !cir.ptr<!void>

int *ip = 0;
// CHECK: cir.global external @ip = #cir.ptr<null> : !cir.ptr<!s32i>

double *dp;
// CHECK: cir.global external @dp : !cir.ptr<!cir.double>

char **cpp;
// CHECK: cir.global external @cpp : !cir.ptr<!cir.ptr<!s8i>>

void (*fp)();
// CHECK: cir.global external @fp : !cir.ptr<!cir.func<()>>

int (*fpii)(int) = 0;
// CHECK: cir.global external @fpii = #cir.ptr<null> : !cir.ptr<!cir.func<(!s32i) -> !s32i>>

void (*fpvar)(int, ...);
// CHECK: cir.global external @fpvar : !cir.ptr<!cir.func<(!s32i, ...)>>
