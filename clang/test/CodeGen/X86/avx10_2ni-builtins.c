// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64 -target-feature +avx10.2 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i686 -target-feature +avx10.2 -emit-llvm -o - | FileCheck %s

#include <immintrin.h>

// VNNI FP16
__m128 test_mm_dpph_ps(__m128 __W, __m128h __A, __m128h __B) {
// CHECK-LABEL: @test_mm_dpph_ps(
// CHECK: call <4 x float> @llvm.x86.avx10.vdpphps.128
  return _mm_dpph_ps(__W, __A, __B);
}

__m128 test_mm_mask_dpph_ps(__m128 __W, __mmask8 __U, __m128h __A, __m128h __B) {
// CHECK-LABEL: @test_mm_mask_dpph_ps(
// CHECK: call <4 x float> @llvm.x86.avx10.vdpphps.128
// CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_dpph_ps(__W, __U, __A, __B);
}

__m128 test_mm_maskz_dpph_ps(__mmask8 __U, __m128 __W, __m128h __A, __m128h __B) {
// CHECK-LABEL: @test_mm_maskz_dpph_ps(
// CHECK: call <4 x float> @llvm.x86.avx10.vdpphps.128
// CHECK: zeroinitializer
// CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_dpph_ps(__U, __W, __A, __B);
}

__m256 test_mm256_dpph_ps(__m256 __W, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_dpph_ps(
// CHECK: call <8 x float> @llvm.x86.avx10.vdpphps.256
  return _mm256_dpph_ps(__W, __A, __B);
}

__m256 test_mm256_mask_dpph_ps(__m256 __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_dpph_ps(
// CHECK: call <8 x float> @llvm.x86.avx10.vdpphps.256
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_dpph_ps(__W, __U,  __A, __B);
}

__m256 test_mm256_maskz_dpph_ps(__mmask8 __U, __m256 __W, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_dpph_ps(
// CHECK: call <8 x float> @llvm.x86.avx10.vdpphps.256
// CHECK: zeroinitializer
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_dpph_ps(__U, __W, __A, __B);
}

// VMPSADBW
__m128i test_mm_mpsadbw_epu8(__m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mpsadbw_epu8
// CHECK: @llvm.x86.sse41.mpsadbw
  return _mm_mpsadbw_epu8(__A, __B, 170);
}

__m128i test_mm_mask_mpsadbw_epu8(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mask_mpsadbw_epu8
// CHECK: @llvm.x86.sse41.mpsadbw
// CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mpsadbw_epu8(__W, __U, __A, __B, 170);
}

__m128i test_mm_maskz_mpsadbw_epu8(__mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_maskz_mpsadbw_epu8
// CHECK: @llvm.x86.sse41.mpsadbw
// CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mpsadbw_epu8(__U, __A, __B, 170);
}

__m256i test_mm256_mpsadbw_epu8(__m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mpsadbw_epu8
// CHECK: @llvm.x86.avx2.mpsadbw
  return _mm256_mpsadbw_epu8(__A, __B, 170);
}

__m256i test_mm256_mask_mpsadbw_epu8(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mask_mpsadbw_epu8
// CHECK: @llvm.x86.avx2.mpsadbw
// CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mpsadbw_epu8(__W, __U, __A, __B, 170);
}

__m256i test_mm256_maskz_mpsadbw_epu8(__mmask16 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_maskz_mpsadbw_epu8
// CHECK: @llvm.x86.avx2.mpsadbw
// CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mpsadbw_epu8(__U, __A, __B, 170);
}

// VNNI INT8
__m128i test_mm_mask_dpbssd_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mask_dpbssd_epi32
// CHECK: @llvm.x86.avx2.vpdpbssd.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbssd_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_dpbssd_epi32(__mmask8 __U, __m128i __W, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_maskz_dpbssd_epi32
// CHECK: @llvm.x86.avx2.vpdpbssd.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbssd_epi32(__U, __W, __A, __B);
}

__m128i test_mm_mask_dpbssds_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mask_dpbssds_epi32
// CHECK: @llvm.x86.avx2.vpdpbssds.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbssds_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_dpbssds_epi32(__mmask8 __U, __m128i __W, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_maskz_dpbssds_epi32
// CHECK: @llvm.x86.avx2.vpdpbssds.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbssds_epi32(__U, __W, __A, __B);
}

__m128i test_mm_mask_dpbsud_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mask_dpbsud_epi32
// CHECK: @llvm.x86.avx2.vpdpbsud.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbsud_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_dpbsud_epi32(__mmask8 __U, __m128i __W, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_maskz_dpbsud_epi32
// CHECK: @llvm.x86.avx2.vpdpbsud.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbsud_epi32(__U, __W, __A, __B);
}

__m128i test_mm_mask_dpbsuds_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mask_dpbsuds_epi32
// CHECK: @llvm.x86.avx2.vpdpbsuds.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbsuds_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_dpbsuds_epi32(__mmask8 __U, __m128i __W, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_maskz_dpbsuds_epi32
// CHECK: @llvm.x86.avx2.vpdpbsuds.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbsuds_epi32(__U, __W, __A, __B);
}

__m128i test_mm_mask_dpbuud_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mask_dpbuud_epi32
// CHECK: @llvm.x86.avx2.vpdpbuud.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbuud_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_dpbuud_epi32(__mmask8 __U, __m128i __W, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_maskz_dpbuud_epi32
// CHECK: @llvm.x86.avx2.vpdpbuud.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbuud_epi32(__U, __W, __A, __B);
}

__m128i test_mm_mask_dpbuuds_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mask_dpbuuds_epi32
// CHECK: @llvm.x86.avx2.vpdpbuuds.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpbuuds_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_dpbuuds_epi32(__mmask8 __U, __m128i __W, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_maskz_dpbuuds_epi32
// CHECK: @llvm.x86.avx2.vpdpbuuds.128
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpbuuds_epi32(__U, __W, __A, __B);
}

__m256i test_mm256_mask_dpbssd_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mask_dpbssd_epi32
// CHECK: @llvm.x86.avx2.vpdpbssd.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpbssd_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_dpbssd_epi32(__mmask8 __U, __m256i __W, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_maskz_dpbssd_epi32
// CHECK: @llvm.x86.avx2.vpdpbssd.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbssd_epi32(__U, __W, __A, __B);
}

__m256i test_mm256_mask_dpbssds_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mask_dpbssds_epi32
// CHECK: @llvm.x86.avx2.vpdpbssds.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpbssds_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_dpbssds_epi32(__mmask8 __U, __m256i __W, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_maskz_dpbssds_epi32
// CHECK: @llvm.x86.avx2.vpdpbssds.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbssds_epi32(__U, __W, __A, __B);
}

__m256i test_mm256_mask_dpbsud_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mask_dpbsud_epi32
// CHECK: @llvm.x86.avx2.vpdpbsud.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpbsud_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_dpbsud_epi32(__mmask8 __U, __m256i __W, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_maskz_dpbsud_epi32
// CHECK: @llvm.x86.avx2.vpdpbsud.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbsud_epi32(__U, __W, __A, __B);
}

__m256i test_mm256_mask_dpbsuds_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mask_dpbsuds_epi32
// CHECK: @llvm.x86.avx2.vpdpbsuds.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpbsuds_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_dpbsuds_epi32(__mmask8 __U, __m256i __W, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_maskz_dpbsuds_epi32
// CHECK: @llvm.x86.avx2.vpdpbsuds.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbsuds_epi32(__U, __W, __A, __B);
}

__m256i test_mm256_mask_dpbuud_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mask_dpbuud_epi32
// CHECK: @llvm.x86.avx2.vpdpbuud.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpbuud_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_dpbuud_epi32(__mmask8 __U, __m256i __W, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_maskz_dpbuud_epi32
// CHECK: @llvm.x86.avx2.vpdpbuud.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbuud_epi32(__U, __W, __A, __B);
}

__m256i test_mm256_mask_dpbuuds_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mask_dpbuuds_epi32
// CHECK: @llvm.x86.avx2.vpdpbuuds.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpbuuds_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_dpbuuds_epi32(__mmask8 __U, __m256i __W, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_maskz_dpbuuds_epi32
// CHECK: @llvm.x86.avx2.vpdpbuuds.256
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpbuuds_epi32(__U, __W, __A, __B);
}

// VNNI INT16
__m128i test_mm_mask_dpwsud_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwsud_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwsud.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwsud_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwsud_epi32(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
// CHECK-LABEL: @test_mm_maskz_dpwsud_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwsud.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwsud_epi32(__U, __A, __B, __C);
}

__m256i test_mm256_mask_dpwsud_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwsud_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwsud.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwsud_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwsud_epi32(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_maskz_dpwsud_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwsud.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwsud_epi32(__U, __A, __B, __C);
}

__m128i test_mm_mask_dpwsuds_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwsuds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwsuds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwsuds_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwsuds_epi32(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
// CHECK-LABEL: @test_mm_maskz_dpwsuds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwsuds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwsuds_epi32(__U, __A, __B, __C);
}

__m256i test_mm256_mask_dpwsuds_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwsuds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwsuds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwsuds_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwsuds_epi32(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_maskz_dpwsuds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwsuds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwsuds_epi32(__U, __A, __B, __C);
}

__m128i test_mm_mask_dpwusd_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwusd_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwusd.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwusd_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwusd_epi32(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
// CHECK-LABEL: @test_mm_maskz_dpwusd_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwusd.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwusd_epi32(__U, __A, __B, __C);
}

__m256i test_mm256_mask_dpwusd_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwusd_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwusd.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwusd_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwusd_epi32(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_maskz_dpwusd_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwusd.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwusd_epi32(__U, __A, __B, __C);
}

__m128i test_mm_mask_dpwusds_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwusds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwusds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwusds_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwusds_epi32(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
// CHECK-LABEL: @test_mm_maskz_dpwusds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwusds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwusds_epi32(__U, __A, __B, __C);
}

__m256i test_mm256_mask_dpwusds_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwusds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwusds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwusds_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwusds_epi32(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_maskz_dpwusds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwusds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwusds_epi32(__U, __A, __B, __C);
}

__m128i test_mm_mask_dpwuud_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwuud_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwuud.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwuud_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwuud_epi32(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
// CHECK-LABEL: @test_mm_maskz_dpwuud_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwuud.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwuud_epi32(__U, __A, __B, __C);
}

__m256i test_mm256_mask_dpwuud_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwuud_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwuud.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwuud_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwuud_epi32(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_maskz_dpwuud_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwuud.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwuud_epi32(__U, __A, __B, __C);
}

__m128i test_mm_mask_dpwuuds_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwuuds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwuuds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwuuds_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwuuds_epi32(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
// CHECK-LABEL: @test_mm_maskz_dpwuuds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwuuds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwuuds_epi32(__U, __A, __B, __C);
}

__m256i test_mm256_mask_dpwuuds_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwuuds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwuuds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwuuds_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwuuds_epi32(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_maskz_dpwuuds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwuuds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwuuds_epi32(__U, __A, __B, __C);
}
