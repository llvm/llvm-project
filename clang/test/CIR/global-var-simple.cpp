// Global variables of intergal types
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir

char c;
// CHECK: cir.global external @c = #cir.int<0> : !s8i

signed char sc;
// CHECK: cir.global external @sc = #cir.int<0> : !s8i

unsigned char uc;
// CHECK: cir.global external @uc = #cir.int<0> : !u8i

short ss;
// CHECK: cir.global external @ss = #cir.int<0> : !s16i

unsigned short us;
// CHECK: cir.global external @us = #cir.int<0> : !u16i

int si;
// CHECK: cir.global external @si = #cir.int<0> : !s32i

unsigned ui;
// CHECK: cir.global external @ui = #cir.int<0> : !u32i

long sl;
// CHECK: cir.global external @sl = #cir.int<0> : !s64i

unsigned long ul;
// CHECK: cir.global external @ul = #cir.int<0> : !u64i

long long sll;
// CHECK: cir.global external @sll = #cir.int<0> : !s64i

unsigned long long ull;
// CHECK: cir.global external @ull = #cir.int<0> : !u64i

__int128 s128;
// CHECK: cir.global external @s128 = #cir.int<0> : !s128i

unsigned __int128 u128;
// CHECK: cir.global external @u128 = #cir.int<0> : !u128i

wchar_t wc;
// CHECK: cir.global external @wc = #cir.int<0> : !s32i

char8_t c8;
// CHECK: cir.global external @c8 = #cir.int<0> : !u8i

char16_t c16;
// CHECK: cir.global external @c16 = #cir.int<0> : !u16i

char32_t c32;
// CHECK: cir.global external @c32 = #cir.int<0> : !u32i

_BitInt(20) sb20;
// CHECK: cir.global external @sb20 = #cir.int<0> : !cir.int<s, 20>

unsigned _BitInt(48) ub48;
// CHECK: external @ub48 = #cir.int<0> : !u48i
