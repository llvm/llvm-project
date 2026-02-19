// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding -triple x86_64-unknown-linux-gnu -fclangir -target-feature +avx512vbmi2 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding -triple x86_64-unknown-linux-gnu -fclangir -target-feature +avx512vbmi2 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vbmi2 -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s


#include <immintrin.h>

__m512i test_mm512_shldv_epi64(__m512i s, __m512i a, __m512i b) {
  // CIR-LABEL: @_mm512_shldv_epi64
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<8 x !u64i>
  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !u64i>, !cir.vector<8 x !u64i>, !cir.vector<8 x !u64i>) -> !cir.vector<8 x !u64i>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<8 x !u64i> -> !cir.vector<8 x !s64i>
  // CIR-LABEL: @test_mm512_shldv_epi64
  // CIR: %{{.*}} = cir.call @_mm512_shldv_epi64
  // LLVM-LABEL: @test_mm512_shldv_epi64
  // LLVM: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> {{.*}}, <8 x i64> {{.*}}, <8 x i64>
  // OGCG-LABEL: @test_mm512_shldv_epi64
  // OGCG: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64>
  return _mm512_shldv_epi64(s, a, b);
}

__m512i test_mm512_mask_shldi_epi64(__m512i s, __mmask8 u, __m512i a, __m512i b) {
  // CIR-LABEL: test_mm512_mask_shldi_epi64
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}}
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}})
  // LLVM-LABEL: @test_mm512_mask_shldi_epi64
  // LLVM: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> {{.*}}, <8 x i64> {{.*}}, <8 x i64> splat (i64 47))
  // LLVM: select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64>
  // OGCG-LABEL: @test_mm512_mask_shldi_epi64
  // OGCG: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 47))
  // OGCG: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shldi_epi64(s, u, a, b, 47);
}

__m512i test_mm512_maskz_shldi_epi64(__mmask8 u, __m512i a, __m512i b) {
  // CIR-LABEL: test_mm512_maskz_shldi_epi64
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !u64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_maskz_shldi_epi64
  // LLVM: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> {{.*}}, <8 x i64> {{.*}}, <8 x i64> splat (i64 63))
  // LLVM: select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64>
  // OGCG-LABEL: @test_mm512_maskz_shldi_epi64
  // OGCG: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 63))
  // OGCG: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shldi_epi64(u, a, b, 63);
}

__m512i test_mm512_shldi_epi64(__m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_shldi_epi64
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !u64i>) -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_shldi_epi64
  // LLVM: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> {{.*}}, <8 x i64> {{.*}}, <8 x i64> splat (i64 31))
  // OGCG-LABEL: @test_mm512_shldi_epi64
  // OGCG: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 31))
  return _mm512_shldi_epi64(a, b, 31);
}

__m512i test_mm512_mask_shldi_epi32(__m512i s, __mmask16 u, __m512i a, __m512i b) {
  // CIR-LABEL: test_mm512_mask_shldi_epi32
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !s32i>, !cir.vector<16 x !s32i>, !cir.vector<16 x !u32i>) -> !cir.vector<16 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<16 x !s32i> -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_mask_shldi_epi32
  // LLVM: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32> splat (i32 7))
  // LLVM: select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // OGCG-LABEL: @test_mm512_mask_shldi_epi32
  // OGCG: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 7))
  // OGCG: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shldi_epi32(s, u, a, b, 7);
}

__m512i test_mm512_maskz_shldi_epi32(__mmask16 u, __m512i a, __m512i b) {
  // CIR-LABEL: test_mm512_maskz_shldi_epi32
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !s32i>, !cir.vector<16 x !s32i>, !cir.vector<16 x !u32i>) -> !cir.vector<16 x !s32i>
  // LLVM-LABEL: @test_mm512_maskz_shldi_epi32
  // LLVM: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32> splat (i32 15))
  // LLVM: select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // OGCG-LABEL: @test_mm512_maskz_shldi_epi32
  // OGCG: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 15))
  // OGCG: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shldi_epi32(u, a, b, 15);
}

__m512i test_mm512_shldi_epi32(__m512i a, __m512i b) {
  // CIR-LABEL: test_mm512_shldi_epi32
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !s32i>, !cir.vector<16 x !s32i>, !cir.vector<16 x !u32i>) -> !cir.vector<16 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<16 x !s32i> -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_shldi_epi32
  // LLVM: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32> splat (i32 31))
  // OGCG-LABEL: @test_mm512_shldi_epi32
  // OGCG: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 31))
  return _mm512_shldi_epi32(a, b, 31);
}

__m512i test_mm512_mask_shldi_epi16(__m512i s, __mmask32 u, __m512i a, __m512i b) {
  // CIR-LABEL: test_mm512_mask_shldi_epi16
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<32 x !s16i>, !cir.vector<32 x !s16i>, !cir.vector<32 x !u16i>) -> !cir.vector<32 x !s16i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<32 x !s16i> -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_mask_shldi_epi16
  // LLVM: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16> splat (i16 3))
  // LLVM: select <32 x i1> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // OGCG-LABEL: @test_mm512_mask_shldi_epi16
  // OGCG: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 3))
  // OGCG: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shldi_epi16(s, u, a, b, 3);
}

__m512i test_mm512_maskz_shldi_epi16(__mmask32 u, __m512i a, __m512i b) {
  // CIR-LABEL: test_mm512_maskz_shldi_epi16
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<32 x !s16i>, !cir.vector<32 x !s16i>, !cir.vector<32 x !u16i>) -> !cir.vector<32 x !s16i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<32 x !s16i> -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_maskz_shldi_epi16
  // LLVM: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16> splat (i16 15))
  // LLVM: select <32 x i1> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // OGCG-LABEL: @test_mm512_maskz_shldi_epi16
  // OGCG: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 15))
  // OGCG: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shldi_epi16(u, a, b, 15);
}

__m512i test_mm512_shldi_epi16(__m512i a, __m512i b) {
  // CIR-LABEL: test_mm512_shldi_epi16
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<32 x !s16i>, !cir.vector<32 x !s16i>, !cir.vector<32 x !u16i>) -> !cir.vector<32 x !s16i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<32 x !s16i> -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_shldi_epi16
  // LLVM: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16> splat (i16 31))
  // OGCG-LABEL: @test_mm512_shldi_epi16
  // OGCG: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 31))
  return _mm512_shldi_epi16(a, b, 31);
}

__m512i test_mm512_mask_shldv_epi64(__m512i s, __mmask8 u, __m512i a, __m512i b) {
  // CIR-LABEL: _mm512_mask_shldv_epi64
  // CIR: cir.call @_mm512_shldv_epi64(%{{.*}}, %{{.*}}, %{{.*}}){{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s64i>
  // CIR-LABEL: test_mm512_mask_shldv_epi64
  // CIR: cir.call @_mm512_mask_shldv_epi64
  // LLVM-LABEL: @test_mm512_mask_shldv_epi64
  // LLVM: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> {{.*}}, <8 x i64> {{.*}}, <8 x i64>
  // LLVM: select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64>
  // OGCG-LABEL: @test_mm512_mask_shldv_epi64
  // OGCG: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64>
  // OGCG: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shldv_epi64(s, u, a, b);
}

__m512i test_mm512_shldv_epi32(__m512i s, __m512i a, __m512i b) {
  // CIR-LABEL: _mm512_shldv_epi32
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !u32i>, !cir.vector<16 x !u32i>, !cir.vector<16 x !u32i>) -> !cir.vector<16 x !u32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<16 x !u32i> -> !cir.vector<8 x !s64i>
  // CIR-LABEL: test_mm512_shldv_epi32
  // CIR: cir.call @_mm512_shldv_epi32
  // LLVM-LABEL: @test_mm512_shldv_epi32
  // LLVM: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // OGCG-LABEL: @test_mm512_shldv_epi32
  // OGCG: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32>
  return _mm512_shldv_epi32(s, a, b);
}

__m512i test_mm512_mask_shldv_epi16(__m512i s, __mmask32 u, __m512i a, __m512i b) {
  // CIR-LABEL: @_mm512_mask_shldv_epi16
  // CIR: cir.call @_mm512_shldv_epi16(%{{.*}}, %{{.*}}, %{{.*}}){{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<32 x !s16i>
  // CIR-LABEL: @test_mm512_mask_shldv_epi16
  // CIR: cir.call @_mm512_mask_shldv_epi16
  // LLVM-LABEL: @test_mm512_mask_shldv_epi16
  // LLVM: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // LLVM: select <32 x i1> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // OGCG-LABEL: @test_mm512_mask_shldv_epi16
  // OGCG: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16>
  // OGCG: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shldv_epi16(s, u, a, b);
}

__m512i test_mm512_maskz_shldv_epi16(__mmask32 u, __m512i s, __m512i a, __m512i b) {
  // CIR-LABEL: _mm512_maskz_shldv_epi16
  // CIR: cir.call @_mm512_shldv_epi16(%{{.*}}, %{{.*}}, %{{.*}}){{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<32 x !s16i>
  // CIR-LABEL: @test_mm512_maskz_shldv_epi16
  // CIR: cir.call @_mm512_maskz_shldv_epi16
  // LLVM-LABEL: @test_mm512_maskz_shldv_epi16
  // LLVM: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // LLVM: select <32 x i1> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // OGCG-LABEL: @test_mm512_maskz_shldv_epi16
  // OGCG: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16>
  // OGCG: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shldv_epi16(u, s, a, b);
}

__m512i test_mm512_shldv_epi16(__m512i s, __m512i a, __m512i b) {
  // CIR-LABEL: _mm512_shldv_epi16
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}}{{.*}} : (!cir.vector<32 x !u16i>, !cir.vector<32 x !u16i>, !cir.vector<32 x !u16i>) -> !cir.vector<32 x !u16i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<32 x !u16i> -> !cir.vector<8 x !s64i>
  // CIR-LABEL: @test_mm512_shldv_epi16
  // CIR: cir.call @_mm512_shldv_epi16
  // LLVM-LABEL: @test_mm512_shldv_epi16
  // LLVM: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // OGCG-LABEL: @test_mm512_shldv_epi16
  // OGCG: call <32 x i16> @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16>
  return _mm512_shldv_epi16(s, a, b);
}

__m512i test_mm512_mask_shrdi_epi64(__m512i s, __mmask8 u, __m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_mask_shrdi_epi64
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !u64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_mask_shrdi_epi64
  // LLVM: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> {{.*}}, <8 x i64> {{.*}}, <8 x i64> splat (i64 47))
  // LLVM: select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64>
  // OGCG-LABEL: @test_mm512_mask_shrdi_epi64
  // OGCG: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 47))
  // OGCG: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shrdi_epi64(s, u, a, b, 47);
}

__m512i test_mm512_maskz_shrdi_epi64(__mmask8 u, __m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_maskz_shrdi_epi64
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !u64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.call @_mm512_setzero_si512() {{.*}} : () -> !cir.vector<8 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_maskz_shrdi_epi64
  // LLVM: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> {{.*}}, <8 x i64> {{.*}}, <8 x i64> splat (i64 63))
  // LLVM: select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64>
  // OGCG-LABEL: @test_mm512_maskz_shrdi_epi64
  // OGCG: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 63))
  // OGCG: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shrdi_epi64(u, a, b, 63);
}

__m512i test_mm512_shrdi_epi64(__m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_shrdi_epi64
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !u64i>) -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_shrdi_epi64
  // LLVM: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> {{.*}}, <8 x i64> {{.*}}, <8 x i64> splat (i64 31))
  // OGCG-LABEL: @test_mm512_shrdi_epi64
  // OGCG: call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> splat (i64 31))
  return _mm512_shrdi_epi64(a, b, 31);
}

__m512i test_mm512_mask_shrdi_epi32(__m512i s, __mmask16 u, __m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_mask_shrdi_epi32
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !s32i>, !cir.vector<16 x !s32i>, !cir.vector<16 x !u32i>) -> !cir.vector<16 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<16 x !s32i> -> !cir.vector<8 x !s64i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !s32i>
  // LLVM-LABEL: @test_mm512_mask_shrdi_epi32
  // LLVM: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32> splat (i32 7))
  // LLVM: select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // OGCG-LABEL: @test_mm512_mask_shrdi_epi32
  // OGCG: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 7))
  // OGCG: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shrdi_epi32(s, u, a, b, 7);
}

__m512i test_mm512_maskz_shrdi_epi32(__mmask16 u, __m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_maskz_shrdi_epi32
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !s32i>, !cir.vector<16 x !s32i>, !cir.vector<16 x !u32i>) -> !cir.vector<16 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<16 x !s32i> -> !cir.vector<8 x !s64i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !s32i>
  // LLVM-LABEL: @test_mm512_maskz_shrdi_epi32
  // LLVM: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32> splat (i32 15))
  // LLVM: select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // OGCG-LABEL: @test_mm512_maskz_shrdi_epi32
  // OGCG: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 15))
  // OGCG: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shrdi_epi32(u, a, b, 15);
}

__m512i test_mm512_shrdi_epi32(__m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_shrdi_epi32
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !s32i>, !cir.vector<16 x !s32i>, !cir.vector<16 x !u32i>) -> !cir.vector<16 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<16 x !s32i> -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_shrdi_epi32
  // LLVM: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32> splat (i32 31))
  // OGCG-LABEL: @test_mm512_shrdi_epi32
  // OGCG: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> splat (i32 31))
  return _mm512_shrdi_epi32(a, b, 31);
}

__m512i test_mm512_mask_shrdi_epi16(__m512i s, __mmask32 u, __m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_mask_shrdi_epi16
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<32 x !s16i>, !cir.vector<32 x !s16i>, !cir.vector<32 x !u16i>) -> !cir.vector<32 x !s16i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<32 x !s16i> -> !cir.vector<8 x !s64i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<32 x !cir.int<s, 1>>, !cir.vector<32 x !s16i>
  // LLVM-LABEL: @test_mm512_mask_shrdi_epi16
  // LLVM: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16> splat (i16 3))
  // LLVM: select <32 x i1> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // OGCG-LABEL: @test_mm512_mask_shrdi_epi16
  // OGCG: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 3))
  // OGCG: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shrdi_epi16(s, u, a, b, 3);
}

__m512i test_mm512_maskz_shrdi_epi16(__mmask32 u, __m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_maskz_shrdi_epi16
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<32 x !s16i>, !cir.vector<32 x !s16i>, !cir.vector<32 x !u16i>) -> !cir.vector<32 x !s16i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<32 x !s16i> -> !cir.vector<8 x !s64i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<32 x !cir.int<s, 1>>, !cir.vector<32 x !s16i>
  // LLVM-LABEL: @test_mm512_maskz_shrdi_epi16
  // LLVM: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16> splat (i16 15))
  // LLVM: select <32 x i1> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // OGCG-LABEL: @test_mm512_maskz_shrdi_epi16
  // OGCG: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 15))
  // OGCG: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shrdi_epi16(u, a, b, 15);
}

__m512i test_mm512_shrdi_epi16(__m512i a, __m512i b) {
  // CIR-LABEL: @test_mm512_shrdi_epi16
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<32 x !s16i>, !cir.vector<32 x !s16i>, !cir.vector<32 x !u16i>) -> !cir.vector<32 x !s16i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<32 x !s16i> -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_shrdi_epi16
  // LLVM: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16> splat (i16 31))
  // OGCG-LABEL: @test_mm512_shrdi_epi16
  // OGCG: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> splat (i16 31))
  return _mm512_shrdi_epi16(a, b, 31);
}

__m512i test_mm512_mask_shldv_epi32(__m512i s, __mmask16 u, __m512i a, __m512i b) {
  // CIR-LABEL: _mm512_mask_shldv_epi32
  // CIR: cir.call @_mm512_shldv_epi32(%{{.*}}, %{{.*}}, %{{.*}}){{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<16 x !s32i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !s32i>
  // CIR-LABEL: test_mm512_mask_shldv_epi32
  // CIR: cir.call @_mm512_mask_shldv_epi32
  // LLVM-LABEL: @test_mm512_mask_shldv_epi32
  // LLVM: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // LLVM: select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // OGCG-LABEL: @test_mm512_mask_shldv_epi32
  // OGCG: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32>
  // OGCG: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shldv_epi32(s, u, a, b);
}

__m512i test_mm512_maskz_shldv_epi32(__mmask16 u, __m512i s, __m512i a, __m512i b) {
  // CIR-LABEL: _mm512_maskz_shldv_epi32
  // CIR: cir.call @_mm512_shldv_epi32(%{{.*}}, %{{.*}}, %{{.*}}){{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<16 x !s32i>
  // CIR: cir.call @_mm512_setzero_si512() {{.*}} : () -> !cir.vector<8 x !s64i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !s32i>
  // CIR-LABEL: test_mm512_maskz_shldv_epi32
  // CIR: cir.call @_mm512_maskz_shldv_epi32
  // LLVM-LABEL: @test_mm512_maskz_shldv_epi32
  // LLVM: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // LLVM: select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // OGCG-LABEL: @test_mm512_maskz_shldv_epi32
  // OGCG: call <16 x i32> @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32>
  // OGCG: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shldv_epi32(u, s, a, b);
}

__m512i test_mm512_mask_shrdv_epi32(__m512i s, __mmask16 u, __m512i a, __m512i b) {
  // CIR-LABEL: @_mm512_shrdv_epi32
  // CIR: cir.call @_mm512_shrdv_epi32(%{{.*}}, %{{.*}}, %{{.*}}){{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<16 x !s32i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !s32i>
  // CIR-LABEL: @test_mm512_mask_shrdv_epi32
  // CIR: cir.call @_mm512_mask_shrdv_epi32
  // LLVM-LABEL: @test_mm512_mask_shrdv_epi32
  // LLVM: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // LLVM: select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // OGCG-LABEL: @test_mm512_mask_shrdv_epi32
  // OGCG: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32>
  // OGCG: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shrdv_epi32(s, u, a, b);
}

__m512i test_mm512_maskz_shrdv_epi32(__mmask16 u, __m512i s, __m512i a, __m512i b) {
  // CIR-LABEL: _mm512_maskz_shrdv_epi32
  // CIR: cir.call @_mm512_shrdv_epi32(%{{.*}}, %{{.*}}, %{{.*}}){{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<16 x !s32i>
  // CIR: cir.call @_mm512_setzero_si512() {{.*}} : () -> !cir.vector<8 x !s64i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !s32i>
  // CIR-LABEL: test_mm512_maskz_shrdv_epi32
  // CIR: cir.call @_mm512_maskz_shrdv_epi32
  // LLVM-LABEL: @test_mm512_maskz_shrdv_epi32
  // LLVM: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // LLVM: select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32>
  // OGCG-LABEL: @test_mm512_maskz_shrdv_epi32
  // OGCG: call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32>
  // OGCG: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shrdv_epi32(u, s, a, b);
}

__m512i test_mm512_mask_shrdv_epi16(__m512i s, __mmask32 u, __m512i a, __m512i b) {
  // CIR-LABEL: _mm512_mask_shrdv_epi16
  // CIR: cir.call @_mm512_shrdv_epi16(%{{.*}}, %{{.*}}, %{{.*}}){{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !cir.vector<8 x !s64i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<32 x !s16i>
  // CIR: cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<32 x !cir.int<s, 1>>, !cir.vector<32 x !s16i>
  // CIR-LABEL: test_mm512_mask_shrdv_epi16
  // CIR: cir.call @_mm512_mask_shrdv_epi16
  // LLVM: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // LLVM: select <32 x i1> {{.*}}, <32 x i16> {{.*}}, <32 x i16>
  // OGCG-LABEL: @test_mm512_mask_shrdv_epi16
  // OGCG: call <32 x i16> @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16>
  // OGCG: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shrdv_epi16(s, u, a, b);
}
