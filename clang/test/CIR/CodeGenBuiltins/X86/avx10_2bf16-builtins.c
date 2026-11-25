// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -fclangir -emit-cir -o %t.cir -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -fclangir -emit-llvm -o %t.ll -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefixes=LLVM,OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefixes=LLVM,OGCG

#include <immintrin.h>

__m128bh test_mm_undefined_pbh(void) {
  // CIR-LABEL: cir.func {{.*}}test_mm_undefined_pbh
  // CIR: call @_mm_undefined_pbh

  // CIR-LABEL: cir.func{{.*}} @_mm_undefined_pbh(
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<2 x !cir.double> -> !cir.vector<8 x !cir.bf16>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.bf16>

  // LLVMCIR-LABEL: @test_mm_undefined_pbh
  // LLVMCIR: ret <8 x bfloat> zeroinitializer

  // OGCG-LABEL: test_mm_undefined_pbh
  // OGCG: ret <8 x bfloat> zeroinitializer
  return _mm_undefined_pbh();
}

__m256bh test_mm256_undefined_pbh(void) {
  // CIR-LABEL: cir.func {{.*}}test_mm256_undefined_pbh
  // CIR: call @_mm256_undefined_pbh

  // CIR-LABEL: cir.func{{.*}} @_mm256_undefined_pbh(
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<4 x !cir.double> -> !cir.vector<16 x !cir.bf16>
  // CIR: cir.return %{{.*}} : !cir.vector<16 x !cir.bf16>

  // LLVMCIR-LABEL: @test_mm256_undefined_pbh
  // LLVMCIR: ret <16 x bfloat> zeroinitializer

  // OGCG-LABEL: test_mm256_undefined_pbh
  // OGCG: ret <16 x bfloat> zeroinitializer
  return _mm256_undefined_pbh();
}

__mmask16 test_mm256_mask_fpclass_pbh_mask(__mmask16 __U, __m256bh __A) {
  // CIR-LABEL: _mm256_mask_fpclass_pbh_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx10.fpclass.bf16.256"
  // CIR: %[[B:.*]] = cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.and %[[A]], %[[B]] : !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[C]] : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: test_mm256_mask_fpclass_pbh_mask
  // LLVM: %[[A:.*]] = call <16 x i1> @llvm.x86.avx10.fpclass.bf16.256
  // LLVM: %[[B:.*]] = bitcast i16 {{.*}} to <16 x i1>
  // LLVM: %[[C:.*]] = and <16 x i1> %[[A]], %[[B]]
  // LLVM: bitcast <16 x i1> %[[C]] to i16

  return _mm256_mask_fpclass_pbh_mask(__U, __A, 4);
}

__mmask16 test_mm256_fpclass_pbh_mask(__m256bh __A) {
  // CIR-LABEL: _mm256_fpclass_pbh_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx10.fpclass.bf16.256"
  // CIR: cir.cast bitcast %[[A]] : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: test_mm256_fpclass_pbh_mask
  // LLVM: %[[A:.*]] = call <16 x i1> @llvm.x86.avx10.fpclass.bf16.256
  // LLVM: bitcast <16 x i1> %[[A]] to i16

  return _mm256_fpclass_pbh_mask(__A, 4);
}

__mmask8 test_mm_mask_fpclass_pbh_mask(__mmask8 __U, __m128bh __A) {
  // CIR-LABEL: _mm_mask_fpclass_pbh_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx10.fpclass.bf16.128"
  // CIR: %[[B:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.and %[[A]], %[[B]] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[C]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm_mask_fpclass_pbh_mask
  // LLVM: %[[A:.*]] = call <8 x i1> @llvm.x86.avx10.fpclass.bf16.128
  // LLVM: %[[B:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // LLVM: %[[C:.*]] = and <8 x i1> %[[A]], %[[B]]
  // LLVM: bitcast <8 x i1> %[[C]] to i8

  return _mm_mask_fpclass_pbh_mask(__U, __A, 4);
}

__mmask8 test_mm_fpclass_pbh_mask(__m128bh __A) {
  // CIR-LABEL: _mm_fpclass_pbh_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx10.fpclass.bf16.128"
  // CIR: cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: test_mm_fpclass_pbh_mask
  // LLVM: %[[A:.*]] = call <8 x i1> @llvm.x86.avx10.fpclass.bf16.128
  // LLVM: bitcast <8 x i1> %[[A]] to i8

  return _mm_fpclass_pbh_mask(__A, 4);
}
