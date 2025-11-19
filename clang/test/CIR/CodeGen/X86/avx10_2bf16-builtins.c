// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2-256 -fclangir -emit-cir -o %t.cir -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2-256 -fclangir -emit-llvm -o %t.ll -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

__m128bh test_mm_undefined_pbh(void) {
  // CIR-LABEL: _mm_undefined_pbh
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<2 x !cir.double> -> !cir.vector<8 x !cir.bf16>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.bf16>

  // LLVM-LABEL: @test_mm_undefined_pbh
  // LLVM: store <8 x bfloat> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %{{.*}} = load <8 x bfloat>, ptr %[[A]], align 16
  // LLVM: ret <8 x bfloat> %{{.*}}

  // OGCG-LABEL: test_mm_undefined_pbh
  // OGCG: ret <8 x bfloat> zeroinitializer
  return _mm_undefined_pbh();
}

__m256bh test_mm256_undefined_pbh(void) {
  // CIR-LABEL: _mm256_undefined_pbh
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<4 x !cir.double> -> !cir.vector<16 x !cir.bf16>
  // CIR: cir.return %{{.*}} : !cir.vector<16 x !cir.bf16>

  // LLVM-LABEL: @test_mm256_undefined_pbh
  // LLVM: store <16 x bfloat> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <16 x bfloat>, ptr %[[A]], align 32
  // LLVM: ret <16 x bfloat> %{{.*}}

  // OGCG-LABEL: test_mm256_undefined_pbh
  // OGCG: ret <16 x bfloat> zeroinitializer
  return _mm256_undefined_pbh();
}