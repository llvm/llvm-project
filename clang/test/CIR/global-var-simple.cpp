// Global variables of intergal types
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s

char c;
// CHECK: cir.global @c : !cir.int<s, 8>

signed char sc;
// CHECK: cir.global @sc : !cir.int<s, 8>

unsigned char uc;
// CHECK: cir.global @uc : !cir.int<u, 8>

short ss;
// CHECK: cir.global @ss : !cir.int<s, 16>

unsigned short us;
// CHECK: cir.global @us : !cir.int<u, 16>

int si;
// CHECK: cir.global @si : !cir.int<s, 32>

unsigned ui;
// CHECK: cir.global @ui : !cir.int<u, 32>

long sl;
// CHECK: cir.global @sl : !cir.int<s, 64>

unsigned long ul;
// CHECK: cir.global @ul : !cir.int<u, 64>

long long sll;
// CHECK: cir.global @sll : !cir.int<s, 64>

unsigned long long ull;
// CHECK: cir.global @ull : !cir.int<u, 64>

__int128 s128;
// CHECK: cir.global @s128 : !cir.int<s, 128>

unsigned __int128 u128;
// CHECK: cir.global @u128 : !cir.int<u, 128>

wchar_t wc;
// CHECK: cir.global @wc : !cir.int<s, 32>

char8_t c8;
// CHECK: cir.global @c8 : !cir.int<u, 8>

char16_t c16;
// CHECK: cir.global @c16 : !cir.int<u, 16>

char32_t c32;
// CHECK: cir.global @c32 : !cir.int<u, 32>

_BitInt(20) sb20;
// CHECK: cir.global @sb20 : !cir.int<s, 20>

unsigned _BitInt(48) ub48;
// CHECK: cir.global @ub48 : !cir.int<u, 48>
