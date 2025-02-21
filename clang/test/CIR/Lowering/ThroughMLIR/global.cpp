// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

char c;
// CHECK: memref.global "public" @c : memref<i8>

signed char sc;
// CHECK: memref.global "public" @sc : memref<i8>

unsigned char uc;
// CHECK: memref.global "public" @uc : memref<i8>

short ss;
// CHECK: memref.global "public" @ss : memref<i16>

unsigned short us = 100;
// CHECK: memref.global "public" @us : memref<i16> = dense<100>

int si = 42;
// CHECK: memref.global "public" @si : memref<i32> = dense<42>

unsigned ui;
// CHECK: memref.global "public" @ui : memref<i32>

long sl;
// CHECK: memref.global "public" @sl : memref<i64>

unsigned long ul;
// CHECK: memref.global "public" @ul : memref<i64>

long long sll;
// CHECK: memref.global "public" @sll : memref<i64>

unsigned long long ull = 123456;
// CHECK: memref.global "public" @ull : memref<i64> = dense<123456>

__int128 s128;
// CHECK: memref.global "public" @s128 : memref<i128>

unsigned __int128 u128;
// CHECK: memref.global "public" @u128 : memref<i128>

wchar_t wc;
// CHECK: memref.global "public" @wc : memref<i32>

char8_t c8;
// CHECK: memref.global "public" @c8 : memref<i8>

char16_t c16;
// CHECK: memref.global "public" @c16 : memref<i16>

char32_t c32;
// CHECK: memref.global "public" @c32 : memref<i32>

_BitInt(20) sb20;
// CHECK: memref.global "public" @sb20 : memref<i20>

unsigned _BitInt(48) ub48;
// CHECK: memref.global "public" @ub48 : memref<i48>

_Float16 f16;
// CHECK: memref.global "public" @f16 : memref<f16>

__bf16 bf16;
// CHECK: memref.global "public" @bf16 : memref<bf16>

float f;
// CHECK: memref.global "public" @f : memref<f32>

double d = 1.25;
// CHECK: memref.global "public" @d : memref<f64> = dense<1.250000e+00>

long double ld;
// CHECK: memref.global "public" @ld : memref<f80>

__float128 f128;
// CHECK: memref.global "public" @f128 : memref<f128>

// FIXME: Add global pointers when they can be lowered to MLIR
