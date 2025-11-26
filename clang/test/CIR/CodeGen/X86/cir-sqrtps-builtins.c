// Test for x86 sqrt builtins (sqrtps, sqrtpd, sqrtss, sqrtsd, etc.)
// RUN: %clang_cc1 -fcir -triple x86_64-unknown-linux-gnu -O0 %s -emit-cir -o - | FileCheck %s

#include <immintrin.h>

// Test __builtin_ia32_sqrtps - single precision vector sqrt (128-bit)
__m128 test_sqrtps(__m128 x) {
  return __builtin_ia32_sqrtps(x);
}
// CHECK-LABEL: cir.func @test_sqrtps
// CHECK: cir.sqrt

// Test __builtin_ia32_sqrtps256 - single precision vector sqrt (256-bit)
__m256 test_sqrtps256(__m256 x) {
  return __builtin_ia32_sqrtps256(x);
}
// CHECK-LABEL: cir.func @test_sqrtps256
// CHECK: cir.sqrt

// Test __builtin_ia32_sqrtps512 - single precision vector sqrt (512-bit)
__m512 test_sqrtps512(__m512 x) {
  return __builtin_ia32_sqrtps512(x);
}
// CHECK-LABEL: cir.func @test_sqrtps512
// CHECK: cir.sqrt

// Test __builtin_ia32_sqrtpd - double precision vector sqrt (128-bit)
__m128d test_sqrtpd(__m128d x) {
  return __builtin_ia32_sqrtpd(x);
}
// CHECK-LABEL: cir.func @test_sqrtpd
// CHECK: cir.sqrt

// Test __builtin_ia32_sqrtpd256 - double precision vector sqrt (256-bit)
__m256d test_sqrtpd256(__m256d x) {
  return __builtin_ia32_sqrtpd256(x);
}
// CHECK-LABEL: cir.func @test_sqrtpd256
// CHECK: cir.sqrt

// Test __builtin_ia32_sqrtpd512 - double precision vector sqrt (512-bit)
__m512d test_sqrtpd512(__m512d x) {
  return __builtin_ia32_sqrtpd512(x);
}
// CHECK-LABEL: cir.func @test_sqrtpd512
// CHECK: cir.sqrt
