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
// CIR: cir.func @test_sqrtph512
// CIR: [[RES:%.*]] = cir.sqrt {{%.*}} : !cir.vector<!cir.fp16 x 32>
// CIR: cir.return [[RES]]
// LLVM: define {{.*}} @test_sqrtph512
// LLVM: call <32 x half> @llvm.sqrt.v32f16
// OGCG: define {{.*}} @test_sqrtph512
// OGCG: call <32 x half> @llvm.sqrt.v32f16

// Test __builtin_ia32_sqrtps512
__m512 test_sqrtps512(__m512 a) {
  return __builtin_ia32_sqrtps512(a);
}
// CIR: cir.func @test_sqrtps512
// CIR: [[RES:%.*]] = cir.sqrt {{%.*}} : !cir.vector<!cir.float x 16>
// CIR: cir.return [[RES]]
// LLVM: define {{.*}} @test_sqrtps512
// LLVM: call <16 x float> @llvm.sqrt.v16f32
// OGCG: define {{.*}} @test_sqrtps512
// OGCG: call <16 x float> @llvm.sqrt.v16f32

// Test __builtin_ia32_sqrtpd512
__m512d test_sqrtpd512(__m512d a) {
  return __builtin_ia32_sqrtpd512(a);
}
// CIR: cir.func @test_sqrtpd512
// CIR: [[RES:%.*]] = cir.sqrt {{%.*}} : !cir.vector<!cir.double x 8>
// CIR: cir.return [[RES]]
// LLVM: define {{.*}} @test_sqrtpd512
// LLVM: call <8 x double> @llvm.sqrt.v8f64
// OGCG: define {{.*}} @test_sqrtpd512
// OGCG: call <8 x double> @llvm.sqrt.v8f64