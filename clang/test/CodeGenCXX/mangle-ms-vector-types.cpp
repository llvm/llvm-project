// RUN: %clang_cc1 -fms-extensions -ffreestanding -target-feature +avx -emit-llvm %s -o - -triple=i686-pc-win32 | FileCheck %s

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

void foo64(__m64) {}
// CHECK: define void @"\01?foo64@@YAXT__m64@@@Z"

void foo128(__m128) {}
// CHECK: define void @"\01?foo128@@YAXT__m128@@@Z"

void foo128d(__m128d) {}
// CHECK: define void @"\01?foo128d@@YAXU__m128d@@@Z"

void foo128i(__m128i) {}
// CHECK: define void @"\01?foo128i@@YAXT__m128i@@@Z"

void foo256(__m256) {}
// CHECK: define void @"\01?foo256@@YAXT__m256@@@Z"

void foo256d(__m256d) {}
// CHECK: define void @"\01?foo256d@@YAXU__m256d@@@Z"

void foo256i(__m256i) {}
// CHECK: define void @"\01?foo256i@@YAXT__m256i@@@Z"

// We have a custom mangling for vector types not standardized by Intel.
void foov8hi(__v8hi) {}
// CHECK: define void @"\01?foov8hi@@YAXT__clang_vec8_F@@@Z"

typedef __attribute__((ext_vector_type(4))) int vi4b;
void foovi4b(vi4b) {}
// CHECK: define void @"\01?foovi4b@@YAXT__clang_vec4_H@@@Z"

// Clang does not support vectors of complex types, so we can't test the
// mangling of them.
