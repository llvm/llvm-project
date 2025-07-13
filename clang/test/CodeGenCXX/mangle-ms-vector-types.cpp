// RUN: %clang_cc1 -fms-extensions -fcxx-exceptions -ffreestanding -target-feature +avx -emit-llvm %s -o - -triple=i686-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -fms-extensions -fcxx-exceptions -ffreestanding -target-feature +avx -emit-llvm %s -o - -triple=i686-pc-win32 -fexperimental-new-constant-interpreter | FileCheck %s

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

void thow(int i) {
  switch (i) {
    case 0: throw __m64();
    // CHECK: ??_R0?AT__m64@@@8
    // CHECK: _CT??_R0?AT__m64@@@88
    // CHECK: _CTA1?AT__m64@@
    // CHECK: _TI1?AT__m64@@
    case 1: throw __m128();
    // CHECK: ??_R0?AT__m128@@@8
    // CHECK: _CT??_R0?AT__m128@@@816
    // CHECK: _CTA1?AT__m128@@
    // CHECK: _TI1?AT__m128@@
    case 2: throw __m128d();
    // CHECK: ??_R0?AU__m128d@@@8
    // CHECK: _CT??_R0?AU__m128d@@@816
    // CHECK: _CTA1?AU__m128d@@
    // CHECK: _TI1?AU__m128d@@
    case 3: throw __m128i();
    // CHECK: ??_R0?AT__m128i@@@8
    // CHECK: _CT??_R0?AT__m128i@@@816
    // CHECK: _CTA1?AT__m128i@@
    // CHECK: _TI1?AT__m128i@@
    case 4: throw __m256();
    // CHECK: ??_R0?AT__m256@@@8
    // CHECK: _CT??_R0?AT__m256@@@832
    // CHECK: _CTA1?AT__m256@@
    // CHECK: _TI1?AT__m256@@
    case 5: throw __m256d();
    // CHECK: ??_R0?AU__m256d@@@8
    // CHECK: _CT??_R0?AU__m256d@@@832
    // CHECK: _CTA1?AU__m256d@@
    // CHECK: _TI1?AU__m256d@@
    case 6: throw __m256i();
    // CHECK: ??_R0?AT__m256@@@8
    // CHECK: _CT??_R0?AT__m256@@@832
    // CHECK: _CTA1?AT__m256@@
    // CHECK: _TI1?AT__m256@@
  }
}

void foo64(__m64) {}
// CHECK: define dso_local void @"?foo64@@YAXT__m64@@@Z"

__m64 rfoo64() { return __m64(); }
// CHECK: define dso_local noundef <1 x i64> @"?rfoo64@@YA?AT__m64@@XZ"

void foo128(__m128) {}
// CHECK: define dso_local void @"?foo128@@YAXT__m128@@@Z"

const __m128 rfoo128() { return __m128(); }
// CHECK: define dso_local noundef <4 x float> @"?rfoo128@@YA?BT__m128@@XZ"

void foo128d(__m128d) {}
// CHECK: define dso_local void @"?foo128d@@YAXU__m128d@@@Z"

volatile __m128d rfoo128d() { return __m128d(); }
// CHECK: define dso_local noundef <2 x double> @"?rfoo128d@@YA?CU__m128d@@XZ"

void foo128i(__m128i) {}
// CHECK: define dso_local void @"?foo128i@@YAXT__m128i@@@Z"

const volatile __m128i rfoo128i() { return __m128i(); }
// CHECK: define dso_local noundef <2 x i64> @"?rfoo128i@@YA?DT__m128i@@XZ"

void foo256(__m256) {}
// CHECK: define dso_local void @"?foo256@@YAXT__m256@@@Z"

__m256 rfoo256() { return __m256(); }
// CHECK: define dso_local noundef <8 x float> @"?rfoo256@@YA?AT__m256@@XZ"

void foo256d(__m256d) {}
// CHECK: define dso_local void @"?foo256d@@YAXU__m256d@@@Z"

__m256d rfoo256d() { return __m256d(); }
// CHECK: define dso_local noundef <4 x double> @"?rfoo256d@@YA?AU__m256d@@XZ"

void foo256i(__m256i) {}
// CHECK: define dso_local void @"?foo256i@@YAXT__m256i@@@Z"

__m256i rfoo256i() { return __m256i(); }
// CHECK: define dso_local noundef <4 x i64> @"?rfoo256i@@YA?AT__m256i@@XZ"

// We have a custom mangling for vector types not standardized by Intel.
void foov8hi(__v8hi) {}
// CHECK: define dso_local void @"?foov8hi@@YAXT?$__vector@F$07@__clang@@@Z"

typedef __attribute__((ext_vector_type(4))) int vi4b;
void foovi4b(vi4b) {}
// CHECK: define dso_local void @"?foovi4b@@YAXT?$__vector@H$03@__clang@@@Z"

typedef float __attribute__((__ext_vector_type__(3))) vf3;
void foovf3(vf3) {}
// CHECK: define dso_local void @"?foovf3@@YAXT?$__vector@M$02@__clang@@@Z"

// Clang does not support vectors of complex types, so we can't test the
// mangling of them.
