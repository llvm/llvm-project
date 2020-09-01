// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=i386-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=i386-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <x86intrin.h>

unsigned int test_castf32_u32 (float __A){
  // CHECK-LABEL: test_castf32_u32
  // CHECK: bitcast float* %{{.*}} to i32*
  // CHECK: %{{.*}} = load i32, i32* %{{.*}}, align 4
  return _castf32_u32(__A);
}

unsigned long long test_castf64_u64 (double __A){
  // CHECK-LABEL: test_castf64_u64
  // CHECK: bitcast double* %{{.*}} to i64*
  // CHECK: %{{.*}} = load i64, i64* %{{.*}}, align 8
  return _castf64_u64(__A);
}

float test_castu32_f32 (unsigned int __A){
  // CHECK-LABEL: test_castu32_f32
  // CHECK: bitcast i32* %{{.*}} to float*
  // CHECK: %{{.*}} = load float, float* %{{.*}}, align 4
  return _castu32_f32(__A);
}

double test_castu64_f64 (unsigned long long __A){
  // CHECK-LABEL: test_castu64_f64
  // CHECK: bitcast i64* %{{.*}} to double*
  // CHECK: %{{.*}} = load double, double* %{{.*}}, align 8
  return _castu64_f64(__A);
}

// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)
char cast_f32_u32_0[_castf32_u32(-0.0f) == 0x80000000 ? 1 : -1];
char cast_u32_f32_0[_castu32_f32(0x3F800000) == +1.0f ? 1 : -1];

char castf64_u64_0[_castf64_u64(-0.0) == 0x8000000000000000 ? 1 : -1];
char castu64_f64_0[_castu64_f64(0xBFF0000000000000ULL) == -1.0 ? 1 : -1];
#endif
