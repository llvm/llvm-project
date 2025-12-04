#include <immintrin.h>
// Test X86-specific sqrt builtins

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Test __builtin_ia32_sqrtph512
__m512h test_sqrtph512(__m512h a) {
  return __builtin_ia32_sqrtph512(a);
}
// CHECK: cir.func @test_sqrtph512
// CHECK: [[RES:%.*]] = cir.sqrt {{%.*}} : !cir.vector<!cir.fp16 x 32>
// CHECK: cir.return [[RES]]

// Test __builtin_ia32_sqrtps512
__m512 test_sqrtps512(__m512 a) {
  return __builtin_ia32_sqrtps512(a);
}
// CHECK: cir.func @test_sqrtps512
// CHECK: [[RES:%.*]] = cir.sqrt {{%.*}} : !cir.vector<!cir.float x 16>
// CHECK: cir.return [[RES]]

// Test __builtin_ia32_sqrtpd512
__m512d test_sqrtpd512(__m512d a) {
  return __builtin_ia32_sqrtpd512(a);
}
// CHECK: cir.func @test_sqrtpd512
// CHECK: [[RES:%.*]] = cir.sqrt {{%.*}} : !cir.vector<!cir.double x 8>
// CHECK: cir.return [[RES]]