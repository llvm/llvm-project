// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -fclangir -emit-llvm -o %t.ll  -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

__m128h test_mm_undefined_ph(void) {
  // CIR-LABEL: _mm_undefined_ph
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<2 x !cir.double> -> !cir.vector<8 x !cir.f16>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.f16>

  // LLVM-LABEL: @test_mm_undefined_ph
  // LLVM: store <8 x half> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %{{.*}} = load <8 x half>, ptr %[[A]], align 16
  // LLVM: ret <8 x half> %{{.*}}

  // OGCG-LABEL: test_mm_undefined_ph
  // OGCG: ret <8 x half> zeroinitializer
  return _mm_undefined_ph();
}

__m256h test_mm256_undefined_ph(void) {
  // CIR-LABEL: _mm256_undefined_ph
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<4 x !cir.double> -> !cir.vector<16 x !cir.f16>
  // CIR: cir.return %{{.*}} : !cir.vector<16 x !cir.f16>

  // LLVM-LABEL: @test_mm256_undefined_ph
  // LLVM: store <16 x half> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <16 x half>, ptr %[[A]], align 32
  // LLVM: ret <16 x half> %{{.*}}

  // OGCG-LABEL: test_mm256_undefined_ph
  // OGCG: ret <16 x half> zeroinitializer
  return _mm256_undefined_ph();
}

__m512h test_mm512_undefined_ph(void) {
  // CIR-LABEL: _mm512_undefined_ph
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<32 x !cir.f16>
  // CIR: cir.return %{{.*}} : !cir.vector<32 x !cir.f16>

  // LLVM-LABEL: @test_mm512_undefined_ph
  // LLVM: store <32 x half> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <32 x half>, ptr %[[A]], align 64
  // LLVM: ret <32 x half> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_ph
  // OGCG: ret <32 x half> zeroinitializer
  return _mm512_undefined_ph();
}