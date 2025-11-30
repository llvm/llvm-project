// Test for x86 sqrt builtins (sqrtps, sqrtpd, sqrtph, etc.)
// RUN: %clang_cc1 -fclangir -triple x86_64-unknown-linux-gnu -target-feature +avx512fp16 -emit-cir %s -o - | FileCheck %s

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

// Test __builtin_ia32_sqrtph - half precision vector sqrt (128-bit)
__m128h test_sqrtph(__m128h x) {
  return __builtin_ia32_sqrtph(x);
}
// CHECK-LABEL: cir.func @test_sqrtph
// CHECK: cir.sqrt

// Test __builtin_ia32_sqrtph256 - half precision vector sqrt (256-bit)
__m256h test_sqrtph256(__m256h x) {
  return __builtin_ia32_sqrtph256(x);
}
// CHECK-LABEL: cir.func @test_sqrtph256
// CHECK: cir.sqrt

// Test __builtin_ia32_sqrtph512 - half precision vector sqrt (512-bit)
__m512h test_sqrtph512(__m512h x) {
  return __builtin_ia32_sqrtph512(x);
}
// CHECK-LABEL: cir.func @test_sqrtph512
// CHECK: cir.sqrt