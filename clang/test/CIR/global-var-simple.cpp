// Global variables of scalar types with initial values
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:    -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:    -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:    -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

char c;
// CIR: cir.global external @c = #cir.int<0> : !s8i
// LLVM: @c = global i8 0, align 1
// OGCG: @c = global i8 0, align 1

signed char sc;
// CIR: cir.global external @sc = #cir.int<0> : !s8i
// LLVM: @sc = global i8 0, align 1
// OGCG: @sc = global i8 0, align 1

unsigned char uc;
// CIR: cir.global external @uc = #cir.int<0> : !u8i
// LLVM: @uc = global i8 0, align 1
// OGCG: @uc = global i8 0, align 1

short ss;
// CIR: cir.global external @ss = #cir.int<0> : !s16i
// LLVM: @ss = global i16 0, align 2
// OGCG: @ss = global i16 0, align 2

unsigned short us = 100;
// CIR: cir.global external @us = #cir.int<100> : !u16i
// LLVM: @us = global i16 100, align 2
// OGCG: @us = global i16 100, align 2

int si = 42;
// CIR: cir.global external @si = #cir.int<42> : !s32i
// LLVM: @si = global i32 42, align 4
// OGCG: @si = global i32 42, align 4

unsigned ui;
// CIR: cir.global external @ui = #cir.int<0> : !u32i
// LLVM: @ui = global i32 0, align 4
// OGCG: @ui = global i32 0, align 4

long sl;
// CIR: cir.global external @sl = #cir.int<0> : !s64i
// LLVM: @sl = global i64 0, align 8
// OGCG: @sl = global i64 0, align 8

unsigned long ul;
// CIR: cir.global external @ul = #cir.int<0> : !u64i
// LLVM: @ul = global i64 0, align 8
// OGCG: @ul = global i64 0, align 8

long long sll;
// CIR: cir.global external @sll = #cir.int<0> : !s64i
// LLVM: @sll = global i64 0, align 8
// OGCG: @sll = global i64 0, align 8

unsigned long long ull = 123456;
// CIR: cir.global external @ull = #cir.int<123456> : !u64i
// LLVM: @ull = global i64 123456, align 8
// OGCG: @ull = global i64 123456, align 8

__int128 s128;
// CIR: cir.global external @s128 = #cir.int<0> : !s128i
// LLVM: @s128 = global i128 0, align 16
// OGCG: @s128 = global i128 0, align 16

unsigned __int128 u128;
// CIR: cir.global external @u128 = #cir.int<0> : !u128i
// LLVM: @u128 = global i128 0, align 16
// OGCG: @u128 = global i128 0, align 16

wchar_t wc;
// CIR: cir.global external @wc = #cir.int<0> : !s32i
// LLVM: @wc = global i32 0, align 4
// OGCG: @wc = global i32 0, align 4

char8_t c8;
// CIR: cir.global external @c8 = #cir.int<0> : !u8i
// LLVM: @c8 = global i8 0, align 1
// OGCG: @c8 = global i8 0, align 1

char16_t c16;
// CIR: cir.global external @c16 = #cir.int<0> : !u16i
// LLVM: @c16 = global i16 0, align 2
// OGCG: @c16 = global i16 0, align 2

char32_t c32;
// CIR: cir.global external @c32 = #cir.int<0> : !u32i
// LLVM: @c32 = global i32 0, align 4
// OGCG: @c32 = global i32 0, align 4

// _BitInt uses exact width in LLVM from CIR but promoted storage in OGCG.
_BitInt(20) sb20;
// CIR: cir.global external @sb20 = #cir.int<0> : !cir.int<s, 20, bitint>
// LLVM: @sb20 = global i20 0, align 4
// OGCG: @sb20 = global i32 0, align 4

unsigned _BitInt(48) ub48;
// CIR: cir.global external @ub48 = #cir.int<0> : !cir.int<u, 48, bitint>
// LLVM: @ub48 = global i48 0, align 8
// OGCG: @ub48 = global i64 0, align 8

bool boolfalse = false;
// CIR: cir.global external @boolfalse = #false
// LLVM: @boolfalse = global i8 0, align 1
// OGCG: @boolfalse = global i8 0, align 1

_Float16 f16;
// CIR: cir.global external @f16 = #cir.fp<0.000000e+00> : !cir.f16
// LLVM: @f16 = global half 0xH0000, align 2
// OGCG: @f16 = global half 0xH0000, align 2

__bf16 bf16;
// CIR: cir.global external @bf16 = #cir.fp<0.000000e+00> : !cir.bf16
// LLVM: @bf16 = global bfloat 0xR0000, align 2
// OGCG: @bf16 = global bfloat 0xR0000, align 2

float f;
// CIR: cir.global external @f = #cir.fp<0.000000e+00> : !cir.float
// LLVM: @f = global float 0.000000e+00, align 4
// OGCG: @f = global float 0.000000e+00, align 4

double d = 1.25;
// CIR: cir.global external @d = #cir.fp<1.250000e+00> : !cir.double
// LLVM: @d = global double 1.250000e+00, align 8
// OGCG: @d = global double 1.250000e+00, align 8

long double ld;
// CIR: cir.global external @ld = #cir.fp<0.000000e+00> : !cir.long_double<!cir.f80>
// LLVM: @ld = global x86_fp80 0xK00000000000000000000, align 16
// OGCG: @ld = global x86_fp80 0xK00000000000000000000, align 16

__float128 f128;
// CIR: cir.global external @f128 = #cir.fp<0.000000e+00> : !cir.f128
// LLVM: @f128 = global fp128 0xL00000000000000000000000000000000, align 16
// OGCG: @f128 = global fp128 0xL00000000000000000000000000000000, align 16

void *vp;
// CIR: cir.global external @vp = #cir.ptr<null> : !cir.ptr<!void>
// LLVM: @vp = global ptr null, align 8
// OGCG: @vp = global ptr null, align 8

int *ip = 0;
// CIR: cir.global external @ip = #cir.ptr<null> : !cir.ptr<!s32i>
// LLVM: @ip = global ptr null, align 8
// OGCG: @ip = global ptr null, align 8

double *dp;
// CIR: cir.global external @dp = #cir.ptr<null> : !cir.ptr<!cir.double>
// LLVM: @dp = global ptr null, align 8
// OGCG: @dp = global ptr null, align 8

char **cpp;
// CIR: cir.global external @cpp = #cir.ptr<null> : !cir.ptr<!cir.ptr<!s8i>>
// LLVM: @cpp = global ptr null, align 8
// OGCG: @cpp = global ptr null, align 8

void (*fp)();
// CIR: cir.global external @fp = #cir.ptr<null> : !cir.ptr<!cir.func<()>>
// LLVM: @fp = global ptr null, align 8
// OGCG: @fp = global ptr null, align 8

int (*fpii)(int) = 0;
// CIR: cir.global external @fpii = #cir.ptr<null> : !cir.ptr<!cir.func<(!s32i) -> !s32i>>
// LLVM: @fpii = global ptr null, align 8
// OGCG: @fpii = global ptr null, align 8

void (*fpvar)(int, ...);
// CIR: cir.global external @fpvar = #cir.ptr<null> : !cir.ptr<!cir.func<(!s32i, ...)>>
// LLVM: @fpvar = global ptr null, align 8
// OGCG: @fpvar = global ptr null, align 8
