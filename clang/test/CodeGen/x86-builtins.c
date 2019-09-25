// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=CHECK-64
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=CHECK-32

#include <x86intrin.h>

unsigned int test_castf32_u32 (float __A){
  // CHECK-64-LABEL: @test_castf32_u32
  // CHECK-64: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{.*}}, i8* align 4 %{{.*}}, i64 4, i1 false)
  // CHECK-64: %{{.*}} = load i32, i32* %{{.*}}, align 4
  // CHECK-32-LABEL: @test_castf32_u32
  // CHECK-32: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %{{.*}}, i8* align 4 %{{.*}}, i32 4, i1 false)
  // CHECK-32: %{{.*}} = load i32, i32* %{{.*}}, align 4
  return _castf32_u32(__A);
}

unsigned long long test_castf64_u64 (double __A){
  // CHECK-64-LABEL: @test_castf64_u64
  // CHECK-64: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %{{.*}}, i8* align 8 %{{.*}}, i64 8, i1 false)
  // CHECK-64: %{{.*}} = load i64, i64* %{{.*}}, align 8
  // CHECK-32-LABEL: @test_castf64_u64
  // CHECK-32: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %{{.*}}, i8* align 8 %{{.*}}, i32 8, i1 false)
  // CHECK-32: %{{.*}} = load i64, i64* %{{.*}}, align 8
  return _castf64_u64(__A);
}

float test_castu32_f32 (unsigned int __A){
  // CHECK-64-LABEL: @test_castu32_f32
  // CHECK-64: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{.*}}, i8* align 4 %{{.*}}, i64 4, i1 false)
  // CHECK-64: %{{.*}} = load float, float* %{{.*}}, align 4
  // CHECK-32-LABEL: @test_castu32_f32
  // CHECK-32: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %{{.*}}, i8* align 4 %{{.*}}, i32 4, i1 false)
  // CHECK-32: %{{.*}} = load float, float* %{{.*}}, align 4
  return _castu32_f32(__A);
}

double test_castu64_f64 (unsigned long long __A){
  // CHECK-64-LABEL: @test_castu64_f64
  // CHECK-64: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %{{.*}}, i8* align 8 %{{.*}}, i64 8, i1 false)
  // CHECK-64: %{{.*}} = load double, double* %{{.*}}, align 8
  // CHECK-32-LABEL: @test_castu64_f64
  // CHECK-32: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %{{.*}}, i8* align 8 %{{.*}}, i32 8, i1 false)
  // CHECK-32: %{{.*}} = load double, double* %{{.*}}, align 8
  return _castu64_f64(__A);
}

