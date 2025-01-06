// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64 -target-feature +avx10.2-256 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i686 -target-feature +avx10.2-256 -emit-llvm -o - | FileCheck %s

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

__m128i test_mm_maskz_dpwsud_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_maskz_dpwsud_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwsud.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwsud_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_mask_dpwsud_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwsud_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwsud.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwsud_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwsud_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_maskz_dpwsud_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwsud.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwsud_epi32(__A, __B, __C, __D);
}

__m128i test_mm_mask_dpwsuds_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwsuds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwsuds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwsuds_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwsuds_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_maskz_dpwsuds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwsuds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwsuds_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_mask_dpwsuds_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwsuds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwsuds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwsuds_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwsuds_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_maskz_dpwsuds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwsuds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwsuds_epi32(__A, __B, __C, __D);
}

__m128i test_mm_mask_dpwusd_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwusd_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwusd.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwusd_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwusd_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_maskz_dpwusd_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwusd.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwusd_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_mask_dpwusd_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwusd_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwusd.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwusd_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwusd_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_maskz_dpwusd_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwusd.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwusd_epi32(__A, __B, __C, __D);
}

__m128i test_mm_mask_dpwusds_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwusds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwusds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwusds_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwusds_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_maskz_dpwusds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwusds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwusds_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_mask_dpwusds_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwusds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwusds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwusds_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwusds_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_maskz_dpwusds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwusds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwusds_epi32(__A, __B, __C, __D);
}

__m128i test_mm_mask_dpwuud_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwuud_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwuud.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwuud_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwuud_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_maskz_dpwuud_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwuud.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwuud_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_mask_dpwuud_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwuud_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwuud.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwuud_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwuud_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_maskz_dpwuud_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwuud.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwuud_epi32(__A, __B, __C, __D);
}

__m128i test_mm_mask_dpwuuds_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_mask_dpwuuds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwuuds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_dpwuuds_epi32(__A, __B, __C, __D);
}

__m128i test_mm_maskz_dpwuuds_epi32(__m128i __A, __mmask8 __B, __m128i __C, __m128i __D) {
// CHECK-LABEL: @test_mm_maskz_dpwuuds_epi32(
// CHECK: call <4 x i32> @llvm.x86.avx2.vpdpwuuds.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_dpwuuds_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_mask_dpwuuds_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_mask_dpwuuds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwuuds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_dpwuuds_epi32(__A, __B, __C, __D);
}

__m256i test_mm256_maskz_dpwuuds_epi32(__m256i __A, __mmask8 __B, __m256i __C, __m256i __D) {
// CHECK-LABEL: @test_mm256_maskz_dpwuuds_epi32(
// CHECK: call <8 x i32> @llvm.x86.avx2.vpdpwuuds.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_dpwuuds_epi32(__A, __B, __C, __D);
}

// YMM Rounding
__m256d test_mm256_add_round_pd(__m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_add_round_pd
// CHECK: @llvm.x86.avx10.vaddpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 11)
  return _mm256_add_round_pd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_add_round_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mask_add_round_pd
// CHECK: @llvm.x86.avx10.vaddpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 10)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_add_round_pd(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_add_round_pd(__mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_maskz_add_round_pd
// CHECK: @llvm.x86.avx10.vaddpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 9)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_add_round_pd(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_add_round_ph(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_add_round_ph
// CHECK: @llvm.x86.avx10.vaddph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 11)
  return _mm256_add_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_add_round_ph(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_add_round_ph
// CHECK: @llvm.x86.avx10.vaddph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 10)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_add_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_add_round_ph(__mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_add_round_ph
// CHECK: @llvm.x86.avx10.vaddph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 9)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_add_round_ph(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_add_round_ps(__m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_add_round_ps
// CHECK: @llvm.x86.avx10.vaddps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 11)
  return _mm256_add_round_ps(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_add_round_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mask_add_round_ps
// CHECK: @llvm.x86.avx10.vaddps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 10)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_add_round_ps(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_add_round_ps(__mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_maskz_add_round_ps
// CHECK: @llvm.x86.avx10.vaddps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 9)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_add_round_ps(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__mmask8 test_mm256_cmp_round_pd_mask(__m256d a, __m256d b) {
// CHECK-LABEL: @test_mm256_cmp_round_pd_mask
// CHECK: fcmp oeq <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_round_pd_mask(a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm256_mask_cmp_round_pd_mask(__mmask8 m, __m256d a, __m256d b) {
// CHECK-LABEL: @test_mm256_mask_cmp_round_pd_mask
// CHECK: [[CMP:%.*]] = fcmp oeq <4 x double> %{{.*}}, %{{.*}}
// CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_round_pd_mask(m, a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask16 test_mm256_cmp_round_ph_mask(__m256h a, __m256h b) {
// CHECK-LABEL: @test_mm256_cmp_round_ph_mask
// CHECK: fcmp oeq <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_round_ph_mask(a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask16 test_mm256_mask_cmp_round_ph_mask(__mmask16 m, __m256h a, __m256h b) {
// CHECK-LABEL: @test_mm256_mask_cmp_round_ph_mask
// CHECK: [[CMP:%.*]] = fcmp oeq <16 x half> %{{.*}}, %{{.*}}
// CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_round_ph_mask(m, a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm256_cmp_round_ps_mask(__m256 a, __m256 b) {
// CHECK-LABEL: @test_mm256_cmp_round_ps_mask
// CHECK: fcmp oeq <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_round_ps_mask(a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm256_mask_cmp_round_ps_mask(__mmask8 m, __m256 a, __m256 b) {
// CHECK-LABEL: @test_mm256_mask_cmp_round_ps_mask
// CHECK: [[CMP:%.*]] = fcmp oeq <8 x float> %{{.*}}, %{{.*}}
// CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_round_ps_mask(m, a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__m128h test_mm256_cvt_roundepi32_ph(__m256i A) {
// CHECK-LABEL: test_mm256_cvt_roundepi32_ph
// CHECK: @llvm.x86.avx512.sitofp.round.v8f16.v8i32(<8 x i32> %{{.*}}, i32 11)
  return _mm256_cvt_roundepi32_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_mask_cvt_roundepi32_ph(__m128h A, __mmask8 B, __m256i C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundepi32_ph
// CHECK: @llvm.x86.avx512.sitofp.round.v8f16.v8i32(<8 x i32> %{{.*}}, i32 10)
// CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm256_mask_cvt_roundepi32_ph(A, B, C, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_maskz_cvt_roundepi32_ph(__mmask8 A, __m256i B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundepi32_ph
// CHECK: @llvm.x86.avx512.sitofp.round.v8f16.v8i32(<8 x i32> %{{.*}}, i32 9)
// CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm256_maskz_cvt_roundepi32_ph(A, B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_cvt_roundepi32_ps(__m256i __A)
{
// CHECK-LABEL: @test_mm256_cvt_roundepi32_ps
// CHECK: @llvm.x86.avx512.sitofp.round.v8f32.v8i32
  return _mm256_cvt_roundepi32_ps(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_cvt_roundepi32_ps(__m256 __W, __mmask8 __U, __m256i __A)
{
// CHECK-LABEL: @test_mm256_mask_cvt_roundepi32_ps
// CHECK: @llvm.x86.avx512.sitofp.round.v8f32.v8i32
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_cvt_roundepi32_ps(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_cvt_roundepi32_ps(__mmask8 __U, __m256i __A)
{
// CHECK-LABEL: @test_mm256_maskz_cvt_roundepi32_ps
// CHECK: @llvm.x86.avx512.sitofp.round.v8f32.v8i32
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_cvt_roundepi32_ps(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_cvt_roundpd_epi32(__m256d A)
{
// CHECK-LABEL: @test_mm256_cvt_roundpd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvtpd2dq256
  return _mm256_cvt_roundpd_epi32(A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_mask_cvt_roundpd_epi32(__m128i W,__mmask8 U,__m256d A)
{
// CHECK-LABEL: @test_mm256_mask_cvt_roundpd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvtpd2dq256
  return _mm256_mask_cvt_roundpd_epi32(W, U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_maskz_cvt_roundpd_epi32(__mmask8 U, __m256d A)
{
// CHECK-LABEL: @test_mm256_maskz_cvt_roundpd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvtpd2dq256
  return _mm256_maskz_cvt_roundpd_epi32(U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_cvt_roundpd_ph(__m256d A) {
// CHECK-LABEL: test_mm256_cvt_roundpd_ph
// CHECK: @llvm.x86.avx10.mask.vcvtpd2ph256
  return _mm256_cvt_roundpd_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_mask_cvt_roundpd_ph(__m128h A, __mmask8 B, __m256d C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundpd_ph
// CHECK: @llvm.x86.avx10.mask.vcvtpd2ph256
  return _mm256_mask_cvt_roundpd_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_maskz_cvt_roundpd_ph(__mmask8 A, __m256d B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundpd_ph
// CHECK: @llvm.x86.avx10.mask.vcvtpd2ph256
  return _mm256_maskz_cvt_roundpd_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm256_cvt_roundpd_ps(__m256d A)
{
// CHECK-LABEL: @test_mm256_cvt_roundpd_ps
// CHECK: @llvm.x86.avx10.mask.vcvtpd2ps256
  return _mm256_cvt_roundpd_ps(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm256_mask_cvt_roundpd_ps(__m128 W, __mmask8 U,__m256d A)
{
// CHECK-LABEL: @test_mm256_mask_cvt_roundpd_ps
// CHECK: @llvm.x86.avx10.mask.vcvtpd2ps256
  return _mm256_mask_cvt_roundpd_ps(W, U, A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm256_maskz_cvt_roundpd_ps(__mmask8 U, __m256d A)
{
// CHECK-LABEL: @test_mm256_maskz_cvt_roundpd_ps
// CHECK: @llvm.x86.avx10.mask.vcvtpd2ps256
  return _mm256_maskz_cvt_roundpd_ps(U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundpd_epi64(__m256d __A) {
// CHECK-LABEL: @test_mm256_cvt_roundpd_epi64
// CHECK: @llvm.x86.avx10.mask.vcvtpd2qq256
  return _mm256_cvt_roundpd_epi64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundpd_epi64(__m256i __W, __mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_mask_cvt_roundpd_epi64
// CHECK: @llvm.x86.avx10.mask.vcvtpd2qq256
  return _mm256_mask_cvt_roundpd_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundpd_epi64(__mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_maskz_cvt_roundpd_epi64
// CHECK: @llvm.x86.avx10.mask.vcvtpd2qq256
  return _mm256_maskz_cvt_roundpd_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_cvt_roundpd_epu32(__m256d A)
{
// CHECK-LABEL: @test_mm256_cvt_roundpd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvtpd2udq256
  return _mm256_cvt_roundpd_epu32(A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_mask_cvt_roundpd_epu32(__m128i W,__mmask8 U,__m256d A)
{
// CHECK-LABEL: @test_mm256_mask_cvt_roundpd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvtpd2udq256
  return _mm256_mask_cvt_roundpd_epu32(W, U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_maskz_cvt_roundpd_epu32(__mmask8 U, __m256d A)
{
// CHECK-LABEL: @test_mm256_maskz_cvt_roundpd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvtpd2udq256
  return _mm256_maskz_cvt_roundpd_epu32(U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundpd_epu64(__m256d __A) {
// CHECK-LABEL: @test_mm256_cvt_roundpd_epu64
// CHECK: @llvm.x86.avx10.mask.vcvtpd2uqq256
  return _mm256_cvt_roundpd_epu64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundpd_epu64(__m256i __W, __mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_mask_cvt_roundpd_epu64
// CHECK: @llvm.x86.avx10.mask.vcvtpd2uqq256
  return _mm256_mask_cvt_roundpd_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundpd_epu64(__mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_maskz_cvt_roundpd_epu64
// CHECK: @llvm.x86.avx10.mask.vcvtpd2uqq256
  return _mm256_maskz_cvt_roundpd_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundph_epi32(__m128h A) {
// CHECK-LABEL: test_mm256_cvt_roundph_epi32
// CHECK: @llvm.x86.avx10.mask.vcvtph2dq256
  return _mm256_cvt_roundph_epi32(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundph_epi32(__m256i A, __mmask16 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundph_epi32
// CHECK: @llvm.x86.avx10.mask.vcvtph2dq256
  return _mm256_mask_cvt_roundph_epi32(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundph_epi32(__mmask16 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundph_epi32
// CHECK: @llvm.x86.avx10.mask.vcvtph2dq256
  return _mm256_maskz_cvt_roundph_epi32(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_cvt_roundph_pd(__m128h A) {
// CHECK-LABEL: test_mm256_cvt_roundph_pd
// CHECK: @llvm.x86.avx10.mask.vcvtph2pd256
  return _mm256_cvt_roundph_pd(A, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_cvt_roundph_pd(__m256d A, __mmask8 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundph_pd
// CHECK: @llvm.x86.avx10.mask.vcvtph2pd256
  return _mm256_mask_cvt_roundph_pd(A, B, C, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_cvt_roundph_pd(__mmask8 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundph_pd
// CHECK: @llvm.x86.avx10.mask.vcvtph2pd256
  return _mm256_maskz_cvt_roundph_pd(A, B, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_cvtx_roundph_ps(__m128h A) {
// CHECK-LABEL: test_mm256_cvtx_roundph_ps
// CHECK: @llvm.x86.avx10.mask.vcvtph2psx256
  return _mm256_cvtx_roundph_ps(A, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_cvtx_roundph_ps(__m256 A, __mmask16 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvtx_roundph_ps
// CHECK: @llvm.x86.avx10.mask.vcvtph2psx256
  return _mm256_mask_cvtx_roundph_ps(A, B, C, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_cvtx_roundph_ps(__mmask16 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvtx_roundph_ps
// CHECK: @llvm.x86.avx10.mask.vcvtph2psx256
  return _mm256_maskz_cvtx_roundph_ps(A, B, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundph_epi64(__m128h A) {
// CHECK-LABEL: test_mm256_cvt_roundph_epi64
// CHECK: @llvm.x86.avx10.mask.vcvtph2qq256
  return _mm256_cvt_roundph_epi64(A, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundph_epi64(__m256i A, __mmask8 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundph_epi64
// CHECK: @llvm.x86.avx10.mask.vcvtph2qq256
  return _mm256_mask_cvt_roundph_epi64(A, B, C, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundph_epi64(__mmask8 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundph_epi64
// CHECK: @llvm.x86.avx10.mask.vcvtph2qq256
  return _mm256_maskz_cvt_roundph_epi64(A, B, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundph_epu32(__m128h A) {
// CHECK-LABEL: test_mm256_cvt_roundph_epu32
// CHECK: @llvm.x86.avx10.mask.vcvtph2udq256
  return _mm256_cvt_roundph_epu32(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundph_epu32(__m256i A, __mmask16 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundph_epu32
// CHECK: @llvm.x86.avx10.mask.vcvtph2udq256
  return _mm256_mask_cvt_roundph_epu32(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundph_epu32(__mmask16 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundph_epu32
// CHECK: @llvm.x86.avx10.mask.vcvtph2udq256
  return _mm256_maskz_cvt_roundph_epu32(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundph_epu64(__m128h A) {
// CHECK-LABEL: test_mm256_cvt_roundph_epu64
// CHECK: @llvm.x86.avx10.mask.vcvtph2uqq256
  return _mm256_cvt_roundph_epu64(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundph_epu64(__m256i A, __mmask8 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundph_epu64
// CHECK: @llvm.x86.avx10.mask.vcvtph2uqq256
  return _mm256_mask_cvt_roundph_epu64(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundph_epu64(__mmask8 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundph_epu64
// CHECK: @llvm.x86.avx10.mask.vcvtph2uqq256
  return _mm256_maskz_cvt_roundph_epu64(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundph_epu16(__m256h A) {
// CHECK-LABEL: test_mm256_cvt_roundph_epu16
// CHECK: @llvm.x86.avx10.mask.vcvtph2uw256
  return _mm256_cvt_roundph_epu16(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundph_epu16(__m256i A, __mmask32 B, __m256h C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundph_epu16
// CHECK: @llvm.x86.avx10.mask.vcvtph2uw256
  return _mm256_mask_cvt_roundph_epu16(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundph_epu16(__mmask32 A, __m256h B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundph_epu16
// CHECK: @llvm.x86.avx10.mask.vcvtph2uw256
  return _mm256_maskz_cvt_roundph_epu16(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundph_epi16(__m256h A) {
// CHECK-LABEL: test_mm256_cvt_roundph_epi16
// CHECK: @llvm.x86.avx10.mask.vcvtph2w256
  return _mm256_cvt_roundph_epi16(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundph_epi16(__m256i A, __mmask32 B, __m256h C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundph_epi16
// CHECK: @llvm.x86.avx10.mask.vcvtph2w256
  return _mm256_mask_cvt_roundph_epi16(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundph_epi16(__mmask32 A, __m256h B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundph_epi16
// CHECK: @llvm.x86.avx10.mask.vcvtph2w256
  return _mm256_maskz_cvt_roundph_epi16(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundps_epi32(__m256 __A)
{
// CHECK-LABEL: @test_mm256_cvt_roundps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvtps2dq256
  return _mm256_cvt_roundps_epi32(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundps_epi32(__m256i __W,__mmask16 __U,__m256 __A)
{
// CHECK-LABEL: @test_mm256_mask_cvt_roundps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvtps2dq256
  return _mm256_mask_cvt_roundps_epi32(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundps_epi32(__mmask16 __U, __m256 __A)
{
// CHECK-LABEL: @test_mm256_maskz_cvt_roundps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvtps2dq256
  return _mm256_maskz_cvt_roundps_epi32(__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_cvt_roundps_pd(__m128 __A) {
// CHECK-LABEL: @test_mm256_cvt_roundps_pd
// CHECK: @llvm.x86.avx10.mask.vcvtps2pd256
  return _mm256_cvt_roundps_pd(__A, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_cvt_roundps_pd(__m256d __W, __mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_mask_cvt_roundps_pd
// CHECK: @llvm.x86.avx10.mask.vcvtps2pd256
  return _mm256_mask_cvt_roundps_pd(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_cvt_roundps_pd(__mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_maskz_cvt_roundps_pd
// CHECK: @llvm.x86.avx10.mask.vcvtps2pd256
  return _mm256_maskz_cvt_roundps_pd(__U, __A, _MM_FROUND_NO_EXC);
}

// FIXME: We may change to @llvm.x86.avx10.mask.vcvtps2ph256 in future.
__m128i test_mm256_cvt_roundps_ph(__m256  __A)
{
  // CHECK-LABEL: @test_mm256_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
    return _mm256_cvt_roundps_ph(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_mask_cvt_roundps_ph(__m128i __W , __mmask16 __U, __m256  __A)
{
  // CHECK-LABEL: @test_mm256_mask_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
    return _mm256_mask_cvt_roundps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_maskz_cvt_roundps_ph(__mmask16 __U, __m256  __A)
{
  // CHECK-LABEL: @test_mm256_maskz_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
    return _mm256_maskz_cvt_roundps_ph(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_cvtx_roundps_ph(__m256 A) {
// CHECK-LABEL: test_mm256_cvtx_roundps_ph
// CHECK: @llvm.x86.avx10.mask.vcvtps2phx256
  return _mm256_cvtx_roundps_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_mask_cvtx_roundps_ph(__m128h A, __mmask16 B, __m256 C) {
// CHECK-LABEL: test_mm256_mask_cvtx_roundps_ph
// CHECK: @llvm.x86.avx10.mask.vcvtps2phx256
  return _mm256_mask_cvtx_roundps_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_maskz_cvtx_roundps_ph(__mmask16 A, __m256 B) {
// CHECK-LABEL: test_mm256_maskz_cvtx_roundps_ph
// CHECK: @llvm.x86.avx10.mask.vcvtps2phx256
  return _mm256_maskz_cvtx_roundps_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundps_epi64(__m128 __A) {
// CHECK-LABEL: @test_mm256_cvt_roundps_epi64
// CHECK: @llvm.x86.avx10.mask.vcvtps2qq256
  return _mm256_cvt_roundps_epi64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundps_epi64(__m256i __W, __mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_mask_cvt_roundps_epi64
// CHECK: @llvm.x86.avx10.mask.vcvtps2qq256
  return _mm256_mask_cvt_roundps_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundps_epi64(__mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_maskz_cvt_roundps_epi64
// CHECK: @llvm.x86.avx10.mask.vcvtps2qq256
  return _mm256_maskz_cvt_roundps_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundps_epu32(__m256 __A)
{
// CHECK-LABEL: @test_mm256_cvt_roundps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvtps2udq256
  return _mm256_cvt_roundps_epu32(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundps_epu32(__m256i __W,__mmask16 __U,__m256 __A)
{
// CHECK-LABEL: @test_mm256_mask_cvt_roundps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvtps2udq256
  return _mm256_mask_cvt_roundps_epu32(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundps_epu32(__mmask16 __U,__m256 __A)
{
// CHECK-LABEL: @test_mm256_maskz_cvt_roundps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvtps2udq256
  return _mm256_maskz_cvt_roundps_epu32(__U,__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvt_roundps_epu64(__m128 __A) {
// CHECK-LABEL: @test_mm256_cvt_roundps_epu64
// CHECK: @llvm.x86.avx10.mask.vcvtps2uqq256
  return _mm256_cvt_roundps_epu64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvt_roundps_epu64(__m256i __W, __mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_mask_cvt_roundps_epu64
// CHECK: @llvm.x86.avx10.mask.vcvtps2uqq256
  return _mm256_mask_cvt_roundps_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvt_roundps_epu64(__mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_maskz_cvt_roundps_epu64
// CHECK: @llvm.x86.avx10.mask.vcvtps2uqq256
  return _mm256_maskz_cvt_roundps_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256d test__mm256_cvt_roundepi64_pd(__m256i __A) {
// CHECK-LABEL: @test__mm256_cvt_roundepi64_pd
// CHECK: @llvm.x86.avx512.sitofp.round.v4f64.v4i64
  return _mm256_cvt_roundepi64_pd(__A, _MM_FROUND_NO_EXC);
}

__m256d test__mm256_mask_cvt_roundepi64_pd(__m256d __W, __mmask8 __U, __m256i __A) {
// CHECK-LABEL: @test__mm256_mask_cvt_roundepi64_pd
// CHECK: @llvm.x86.avx512.sitofp.round.v4f64.v4i64
  return _mm256_mask_cvt_roundepi64_pd(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m256d test__mm256_maskz_cvt_roundepi64_pd(__mmask8 __U, __m256i __A) {
// CHECK-LABEL: @test__mm256_maskz_cvt_roundepi64_pd
// CHECK: @llvm.x86.avx512.sitofp.round.v4f64.v4i64
  return _mm256_maskz_cvt_roundepi64_pd(__U, __A, _MM_FROUND_NO_EXC);
}

// FIXME: We may change to @llvm.x86.avx10.mask.vcvtqq2ph256 in future.
__m128h test_mm256_cvt_roundepi64_ph(__m256i A) {
// CHECK-LABEL: test_mm256_cvt_roundepi64_ph
// CHECK: @llvm.x86.avx512.sitofp.round.v8f16.v4i64
  return _mm256_cvt_roundepi64_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_mask_cvt_roundepi64_ph(__m128h A, __mmask8 B, __m256i C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundepi64_ph
// CHECK: @llvm.x86.avx512.sitofp.round.v8f16.v4i64
  return _mm256_mask_cvt_roundepi64_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_maskz_cvt_roundepi64_ph(__mmask8 A, __m256i B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundepi64_ph
// CHECK: @llvm.x86.avx512.sitofp.round.v8f16.v4i64
  return _mm256_maskz_cvt_roundepi64_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm256_cvt_roundepi64_ps(__m256i __A) {
// CHECK-LABEL: @test_mm256_cvt_roundepi64_ps
// CHECK: @llvm.x86.avx512.sitofp.round.v4f32.v4i64
  return _mm256_cvt_roundepi64_ps(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m128 test_mm256_mask_cvt_roundepi64_ps(__m128 __W, __mmask8 __U, __m256i __A) {
// CHECK-LABEL: @test_mm256_mask_cvt_roundepi64_ps
// CHECK: @llvm.x86.avx512.sitofp.round.v4f32.v4i64
// CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm256_mask_cvt_roundepi64_ps(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m128 test_mm256_maskz_cvt_roundepi64_ps(__mmask8 __U, __m256i __A) {
// CHECK-LABEL: @test_mm256_maskz_cvt_roundepi64_ps
// CHECK: @llvm.x86.avx512.sitofp.round.v4f32.v4i64
// CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm256_maskz_cvt_roundepi64_ps(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_cvtt_roundpd_epi32(__m256d A)
{
// CHECK-LABEL: @test_mm256_cvtt_roundpd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dq256
  return _mm256_cvtt_roundpd_epi32(A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_mask_cvtt_roundpd_epi32(__m128i W,__mmask8 U,__m256d A)
{
// CHECK-LABEL: @test_mm256_mask_cvtt_roundpd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dq256
  return _mm256_mask_cvtt_roundpd_epi32(W, U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_maskz_cvtt_roundpd_epi32(__mmask8 U, __m256d A)
{
// CHECK-LABEL: @test_mm256_maskz_cvtt_roundpd_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2dq256
  return _mm256_maskz_cvtt_roundpd_epi32(U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundpd_epi64(__m256d __A) {
// CHECK-LABEL: @test_mm256_cvtt_roundpd_epi64
// CHECK: @llvm.x86.avx10.mask.vcvttpd2qq256
  return _mm256_cvtt_roundpd_epi64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundpd_epi64(__m256i __W, __mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_mask_cvtt_roundpd_epi64
// CHECK: @llvm.x86.avx10.mask.vcvttpd2qq256
  return _mm256_mask_cvtt_roundpd_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundpd_epi64(__mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_maskz_cvtt_roundpd_epi64
// CHECK: @llvm.x86.avx10.mask.vcvttpd2qq256
  return _mm256_maskz_cvtt_roundpd_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_cvtt_roundpd_epu32(__m256d A)
{
// CHECK-LABEL: @test_mm256_cvtt_roundpd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udq256
  return _mm256_cvtt_roundpd_epu32(A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_mask_cvtt_roundpd_epu32(__m128i W,__mmask8 U,__m256d A)
{
// CHECK-LABEL: @test_mm256_mask_cvtt_roundpd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udq256
  return _mm256_mask_cvtt_roundpd_epu32(W, U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_maskz_cvtt_roundpd_epu32(__mmask8 U, __m256d A)
{
// CHECK-LABEL: @test_mm256_maskz_cvtt_roundpd_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttpd2udq256
  return _mm256_maskz_cvtt_roundpd_epu32(U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundpd_epu64(__m256d __A) {
// CHECK-LABEL: @test_mm256_cvtt_roundpd_epu64
// CHECK: @llvm.x86.avx10.mask.vcvttpd2uqq256
  return _mm256_cvtt_roundpd_epu64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundpd_epu64(__m256i __W, __mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_mask_cvtt_roundpd_epu64
// CHECK: @llvm.x86.avx10.mask.vcvttpd2uqq256
  return _mm256_mask_cvtt_roundpd_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundpd_epu64(__mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_maskz_cvtt_roundpd_epu64
// CHECK: @llvm.x86.avx10.mask.vcvttpd2uqq256
  return _mm256_maskz_cvtt_roundpd_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundph_epi32(__m128h A) {
// CHECK-LABEL: test_mm256_cvtt_roundph_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttph2dq256
  return _mm256_cvtt_roundph_epi32(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundph_epi32(__m256i A, __mmask16 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvtt_roundph_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttph2dq256
  return _mm256_mask_cvtt_roundph_epi32(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundph_epi32(__mmask16 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvtt_roundph_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttph2dq256
  return _mm256_maskz_cvtt_roundph_epi32(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundph_epi64(__m128h A) {
// CHECK-LABEL: test_mm256_cvtt_roundph_epi64
// CHECK: @llvm.x86.avx10.mask.vcvttph2qq256
  return _mm256_cvtt_roundph_epi64(A, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundph_epi64(__m256i A, __mmask8 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvtt_roundph_epi64
// CHECK: @llvm.x86.avx10.mask.vcvttph2qq256
  return _mm256_mask_cvtt_roundph_epi64(A, B, C, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundph_epi64(__mmask8 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvtt_roundph_epi64
// CHECK: @llvm.x86.avx10.mask.vcvttph2qq256
  return _mm256_maskz_cvtt_roundph_epi64(A, B, _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundph_epu32(__m128h A) {
// CHECK-LABEL: test_mm256_cvtt_roundph_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttph2udq256
  return _mm256_cvtt_roundph_epu32(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundph_epu32(__m256i A, __mmask16 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvtt_roundph_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttph2udq256
  return _mm256_mask_cvtt_roundph_epu32(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundph_epu32(__mmask16 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvtt_roundph_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttph2udq256
  return _mm256_maskz_cvtt_roundph_epu32(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundph_epu64(__m128h A) {
// CHECK-LABEL: test_mm256_cvtt_roundph_epu64
// CHECK: @llvm.x86.avx10.mask.vcvttph2uqq256
  return _mm256_cvtt_roundph_epu64(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundph_epu64(__m256i A, __mmask8 B, __m128h C) {
// CHECK-LABEL: test_mm256_mask_cvtt_roundph_epu64
// CHECK: @llvm.x86.avx10.mask.vcvttph2uqq256
  return _mm256_mask_cvtt_roundph_epu64(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundph_epu64(__mmask8 A, __m128h B) {
// CHECK-LABEL: test_mm256_maskz_cvtt_roundph_epu64
// CHECK: @llvm.x86.avx10.mask.vcvttph2uqq256
  return _mm256_maskz_cvtt_roundph_epu64(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundph_epu16(__m256h A) {
// CHECK-LABEL: test_mm256_cvtt_roundph_epu16
// CHECK: @llvm.x86.avx10.mask.vcvttph2uw256
  return _mm256_cvtt_roundph_epu16(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundph_epu16(__m256i A, __mmask32 B, __m256h C) {
// CHECK-LABEL: test_mm256_mask_cvtt_roundph_epu16
// CHECK: @llvm.x86.avx10.mask.vcvttph2uw256
  return _mm256_mask_cvtt_roundph_epu16(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundph_epu16(__mmask32 A, __m256h B) {
// CHECK-LABEL: test_mm256_maskz_cvtt_roundph_epu16
// CHECK: @llvm.x86.avx10.mask.vcvttph2uw256
  return _mm256_maskz_cvtt_roundph_epu16(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundph_epi16(__m256h A) {
// CHECK-LABEL: test_mm256_cvtt_roundph_epi16
// CHECK: @llvm.x86.avx10.mask.vcvttph2w256
  return _mm256_cvtt_roundph_epi16(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundph_epi16(__m256i A, __mmask32 B, __m256h C) {
// CHECK-LABEL: test_mm256_mask_cvtt_roundph_epi16
// CHECK: @llvm.x86.avx10.mask.vcvttph2w256
  return _mm256_mask_cvtt_roundph_epi16(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundph_epi16(__mmask32 A, __m256h B) {
// CHECK-LABEL: test_mm256_maskz_cvtt_roundph_epi16
// CHECK: @llvm.x86.avx10.mask.vcvttph2w256
  return _mm256_maskz_cvtt_roundph_epi16(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundps_epi32(__m256 __A)
{
// CHECK-LABEL: @test_mm256_cvtt_roundps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dq256
  return _mm256_cvtt_roundps_epi32(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundps_epi32(__m256i __W,__mmask16 __U,__m256 __A)
{
// CHECK-LABEL: @test_mm256_mask_cvtt_roundps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dq256
  return _mm256_mask_cvtt_roundps_epi32(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundps_epi32(__mmask16 __U, __m256 __A)
{
// CHECK-LABEL: @test_mm256_maskz_cvtt_roundps_epi32
// CHECK: @llvm.x86.avx10.mask.vcvttps2dq256
  return _mm256_maskz_cvtt_roundps_epi32(__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundps_epi64(__m128 __A) {
// CHECK-LABEL: @test_mm256_cvtt_roundps_epi64
// CHECK: @llvm.x86.avx10.mask.vcvttps2qq256
  return _mm256_cvtt_roundps_epi64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundps_epi64(__m256i __W, __mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_mask_cvtt_roundps_epi64
// CHECK: @llvm.x86.avx10.mask.vcvttps2qq256
  return _mm256_mask_cvtt_roundps_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundps_epi64(__mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_maskz_cvtt_roundps_epi64
// CHECK: @llvm.x86.avx10.mask.vcvttps2qq256
  return _mm256_maskz_cvtt_roundps_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundps_epu32(__m256 __A)
{
// CHECK-LABEL: @test_mm256_cvtt_roundps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udq256
  return _mm256_cvtt_roundps_epu32(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundps_epu32(__m256i __W,__mmask16 __U,__m256 __A)
{
// CHECK-LABEL: @test_mm256_mask_cvtt_roundps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udq256
  return _mm256_mask_cvtt_roundps_epu32(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundps_epu32(__mmask16 __U,__m256 __A)
{
// CHECK-LABEL: @test_mm256_maskz_cvtt_roundps_epu32
// CHECK: @llvm.x86.avx10.mask.vcvttps2udq256
  return _mm256_maskz_cvtt_roundps_epu32(__U,__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_cvtt_roundps_epu64(__m128 __A) {
// CHECK-LABEL: @test_mm256_cvtt_roundps_epu64
// CHECK: @llvm.x86.avx10.mask.vcvttps2uqq256
  return _mm256_cvtt_roundps_epu64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_mask_cvtt_roundps_epu64(__m256i __W, __mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_mask_cvtt_roundps_epu64
// CHECK: @llvm.x86.avx10.mask.vcvttps2uqq256
  return _mm256_mask_cvtt_roundps_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm256_maskz_cvtt_roundps_epu64(__mmask8 __U, __m128 __A) {
// CHECK-LABEL: @test_mm256_maskz_cvtt_roundps_epu64
// CHECK: @llvm.x86.avx10.mask.vcvttps2uqq256
  return _mm256_maskz_cvtt_roundps_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_cvt_roundepu32_ph(__m256i A) {
// CHECK-LABEL: test_mm256_cvt_roundepu32_ph
// CHECK: @llvm.x86.avx512.uitofp.round.v8f16.v8i32(<8 x i32> %{{.*}}, i32 11)
  return _mm256_cvt_roundepu32_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_mask_cvt_roundepu32_ph(__m128h A, __mmask8 B, __m256i C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundepu32_ph
// CHECK: @llvm.x86.avx512.uitofp.round.v8f16.v8i32(<8 x i32> %{{.*}}, i32 10)
// CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm256_mask_cvt_roundepu32_ph(A, B, C, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_maskz_cvt_roundepu32_ph(__mmask8 A, __m256i B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundepu32_ph
// CHECK: @llvm.x86.avx512.uitofp.round.v8f16.v8i32(<8 x i32> %{{.*}}, i32 9)
// CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm256_maskz_cvt_roundepu32_ph(A, B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_cvt_roundepu32_ps(__m256i __A)
{
// CHECK-LABEL: @test_mm256_cvt_roundepu32_ps
// CHECK: @llvm.x86.avx512.uitofp.round.v8f32.v8i32
  return _mm256_cvt_roundepu32_ps(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_cvt_roundepu32_ps(__m256 __W, __mmask8 __U, __m256i __A)
{
// CHECK-LABEL: @test_mm256_mask_cvt_roundepu32_ps
// CHECK: @llvm.x86.avx512.uitofp.round.v8f32.v8i32
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_cvt_roundepu32_ps(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_cvt_roundepu32_ps(__mmask8 __U, __m256i __A)
{
// CHECK-LABEL: @test_mm256_maskz_cvt_roundepu32_ps
// CHECK: @llvm.x86.avx512.uitofp.round.v8f32.v8i32
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_cvt_roundepu32_ps(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test__mm256_cvt_roundepu64_pd(__m256i __A) {
// CHECK-LABEL: @test__mm256_cvt_roundepu64_pd
// CHECK: @llvm.x86.avx512.uitofp.round.v4f64.v4i64
  return _mm256_cvt_roundepu64_pd(__A, _MM_FROUND_NO_EXC);
}

__m256d test__mm256_mask_cvt_roundepu64_pd(__m256d __W, __mmask8 __U, __m256i __A) {
// CHECK-LABEL: @test__mm256_mask_cvt_roundepu64_pd
// CHECK: @llvm.x86.avx512.uitofp.round.v4f64.v4i64
  return _mm256_mask_cvt_roundepu64_pd(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m256d test__mm256_maskz_cvt_roundepu64_pd(__mmask8 __U, __m256i __A) {
// CHECK-LABEL: @test__mm256_maskz_cvt_roundepu64_pd
// CHECK: @llvm.x86.avx512.uitofp.round.v4f64.v4i64
  return _mm256_maskz_cvt_roundepu64_pd(__U, __A, _MM_FROUND_NO_EXC);
}

// FIXME: We may change to @llvm.x86.avx10.mask.vcvtuqq2ph256 in future.
__m128h test_mm256_cvt_roundepu64_ph(__m256i A) {
// CHECK-LABEL: test_mm256_cvt_roundepu64_ph
// CHECK: @llvm.x86.avx512.uitofp.round.v8f16.v4i64
  return _mm256_cvt_roundepu64_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_mask_cvt_roundepu64_ph(__m128h A, __mmask8 B, __m256i C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundepu64_ph
// CHECK: @llvm.x86.avx512.uitofp.round.v8f16.v4i64
  return _mm256_mask_cvt_roundepu64_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm256_maskz_cvt_roundepu64_ph(__mmask8 A, __m256i B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundepu64_ph
// CHECK: @llvm.x86.avx512.uitofp.round.v8f16.v4i64
  return _mm256_maskz_cvt_roundepu64_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm256_cvt_roundepu64_ps(__m256i __A) {
// CHECK-LABEL: @test_mm256_cvt_roundepu64_ps
// CHECK: @llvm.x86.avx512.uitofp.round.v4f32.v4i64
  return _mm256_cvt_roundepu64_ps(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m128 test_mm256_mask_cvt_roundepu64_ps(__m128 __W, __mmask8 __U, __m256i __A) {
// CHECK-LABEL: @test_mm256_mask_cvt_roundepu64_ps
// CHECK: @llvm.x86.avx512.uitofp.round.v4f32.v4i64
// CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm256_mask_cvt_roundepu64_ps(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m128 test_mm256_maskz_cvt_roundepu64_ps(__mmask8 __U, __m256i __A) {
// CHECK-LABEL: @test_mm256_maskz_cvt_roundepu64_ps
// CHECK: @llvm.x86.avx512.uitofp.round.v4f32.v4i64
// CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm256_maskz_cvt_roundepu64_ps(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_cvt_roundepi16_ph(__m256i A) {
// CHECK-LABEL: test_mm256_cvt_roundepi16_ph
// CHECK:   @llvm.x86.avx512.sitofp.round.v16f16.v16i16
  return _mm256_cvt_roundepi16_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_cvt_roundepi16_ph(__m256h A, __mmask16 B, __m256i C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundepi16_ph
// CHECK: @llvm.x86.avx512.sitofp.round.v16f16.v16i16
  return _mm256_mask_cvt_roundepi16_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_cvt_roundepi16_ph(__mmask16 A, __m256i B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundepi16_ph
// CHECK: @llvm.x86.avx512.sitofp.round.v16f16.v16i16
  return _mm256_maskz_cvt_roundepi16_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_cvt_roundepu16_ph(__m256i A) {
// CHECK-LABEL: test_mm256_cvt_roundepu16_ph
// CHECK: @llvm.x86.avx512.uitofp.round.v16f16.v16i16
  return _mm256_cvt_roundepu16_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_cvt_roundepu16_ph(__m256h A, __mmask16 B, __m256i C) {
// CHECK-LABEL: test_mm256_mask_cvt_roundepu16_ph
// CHECK: @llvm.x86.avx512.uitofp.round.v16f16.v16i16
  return _mm256_mask_cvt_roundepu16_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_cvt_roundepu16_ph(__mmask16 A, __m256i B) {
// CHECK-LABEL: test_mm256_maskz_cvt_roundepu16_ph
// CHECK: @llvm.x86.avx512.uitofp.round.v16f16.v16i16
  return _mm256_maskz_cvt_roundepu16_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_div_round_pd(__m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_div_round_pd
// CHECK: @llvm.x86.avx10.vdivpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 11)
  return _mm256_div_round_pd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_div_round_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mask_div_round_pd
// CHECK: @llvm.x86.avx10.vdivpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 10)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_div_round_pd(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_div_round_pd(__mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_maskz_div_round_pd
// CHECK: @llvm.x86.avx10.vdivpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 9)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_div_round_pd(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_div_round_ph(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_div_round_ph
// CHECK: @llvm.x86.avx10.vdivph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 11)
  return _mm256_div_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_div_round_ph(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_div_round_ph
// CHECK: @llvm.x86.avx10.vdivph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 10)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_div_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_div_round_ph(__mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_div_round_ph
// CHECK: @llvm.x86.avx10.vdivph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 9)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_div_round_ph(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_div_round_ps(__m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_div_round_ps
// CHECK: @llvm.x86.avx10.vdivps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 11)
  return _mm256_div_round_ps(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_div_round_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mask_div_round_ps
// CHECK: @llvm.x86.avx10.vdivps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 10)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_div_round_ps(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_div_round_ps(__mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_maskz_div_round_ps
// CHECK: @llvm.x86.avx10.vdivps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 9)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_div_round_ps(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_fcmadd_round_pch(__m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_fcmadd_round_pch
// CHECK: @llvm.x86.avx10.mask.vfcmaddcph256
  return _mm256_fcmadd_round_pch(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_fcmadd_round_pch(__m256h __A, __mmask8 __U, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_fcmadd_round_pch
// CHECK: @llvm.x86.avx10.mask.vfcmaddcph256
// CHECK:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fcmadd_round_pch(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask3_fcmadd_round_pch(__m256h __A, __m256h __B, __m256h __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fcmadd_round_pch
// CHECK: @llvm.x86.avx10.mask.vfcmaddcph256
// CHECK-NOT:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fcmadd_round_pch(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_fcmadd_round_pch(__mmask8 __U, __m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_maskz_fcmadd_round_pch
// CHECK: @llvm.x86.avx10.maskz.vfcmaddcph256
  return _mm256_maskz_fcmadd_round_pch(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_cmul_round_pch(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_cmul_round_pch
// CHECK: @llvm.x86.avx10.mask.vfcmulcph256
  return _mm256_cmul_round_pch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_cmul_round_pch(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_cmul_round_pch
// CHECK: @llvm.x86.avx10.mask.vfcmulcph256
  return _mm256_mask_cmul_round_pch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_cmul_round_pch(__mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_cmul_round_pch
// CHECK: @llvm.x86.avx10.mask.vfcmulcph256
  return _mm256_maskz_cmul_round_pch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_fixupimm_round_pd(__m256d __A, __m256d __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_fixupimm_round_pd
// CHECK: @llvm.x86.avx10.mask.vfixupimmpd256
  return _mm256_fixupimm_round_pd(__A, __B, __C, 5, 8);
}

__m256d test_mm256_mask_fixupimm_round_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_mask_fixupimm_round_pd
// CHECK: @llvm.x86.avx10.mask.vfixupimmpd256
  return _mm256_mask_fixupimm_round_pd(__A, __U, __B, __C, 5, 8);
}

__m256d test_mm256_maskz_fixupimm_round_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_maskz_fixupimm_round_pd
// CHECK: @llvm.x86.avx10.maskz.vfixupimmpd256
  return _mm256_maskz_fixupimm_round_pd(__U, __A, __B, __C, 5, 8);
}

__m256 test_mm256_fixupimm_round_ps(__m256 __A, __m256 __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_fixupimm_round_ps
// CHECK: @llvm.x86.avx10.mask.vfixupimmps256
  return _mm256_fixupimm_round_ps(__A, __B, __C, 5, 8);
}

__m256 test_mm256_mask_fixupimm_round_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_mask_fixupimm_round_ps
// CHECK: @llvm.x86.avx10.mask.vfixupimmps256
  return _mm256_mask_fixupimm_round_ps(__A, __U, __B, __C, 5, 8);
}

__m256 test_mm256_maskz_fixupimm_round_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256i __C) {
// CHECK-LABEL: @test_mm256_maskz_fixupimm_round_ps
// CHECK: @llvm.x86.avx10.maskz.vfixupimmps256
  return _mm256_maskz_fixupimm_round_ps(__U, __A, __B, __C, 5, 8);
}

__m256d test_mm256_fmadd_round_pd(__m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_fmadd_round_pd
// CHECK: @llvm.x86.avx10.vfmaddpd256
  return _mm256_fmadd_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_fmadd_round_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_mask_fmadd_round_pd
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fmadd_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask3_fmadd_round_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmadd_round_pd
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fmadd_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_fmadd_round_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_maskz_fmadd_round_pd
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> zeroinitializer
  return _mm256_maskz_fmadd_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_fmsub_round_pd(__m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_fmsub_round_pd
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
  return _mm256_fmsub_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_fmsub_round_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_mask_fmsub_round_pd
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fmsub_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_fmsub_round_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_maskz_fmsub_round_pd
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> zeroinitializer
  return _mm256_maskz_fmsub_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_fnmadd_round_pd(__m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_fnmadd_round_pd
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
  return _mm256_fnmadd_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask3_fnmadd_round_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fnmadd_round_pd
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fnmadd_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_fnmadd_round_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_maskz_fnmadd_round_pd
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> zeroinitializer
  return _mm256_maskz_fnmadd_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_fnmsub_round_pd(__m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_fnmsub_round_pd
// CHECK: fneg <4 x double>
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
  return _mm256_fnmsub_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_fnmsub_round_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_maskz_fnmsub_round_pd
// CHECK: fneg <4 x double>
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> zeroinitializer
  return _mm256_maskz_fnmsub_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_fmadd_round_ph(__m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_fmadd_round_ph
// CHECK: @llvm.x86.avx10.vfmaddph256
  return _mm256_fmadd_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_fmadd_round_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_fmadd_round_ph
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fmadd_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask3_fmadd_round_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmadd_round_ph
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fmadd_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_fmadd_round_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_maskz_fmadd_round_ph
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> zeroinitializer
  return _mm256_maskz_fmadd_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_fmsub_round_ph(__m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_fmsub_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
  return _mm256_fmsub_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_fmsub_round_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_fmsub_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fmsub_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_fmsub_round_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_maskz_fmsub_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> zeroinitializer
  return _mm256_maskz_fmsub_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_fnmadd_round_ph(__m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_fnmadd_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
  return _mm256_fnmadd_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask3_fnmadd_round_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
// CHECK-LABEL: @test_mm256_mask3_fnmadd_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fnmadd_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_fnmadd_round_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_maskz_fnmadd_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> zeroinitializer
  return _mm256_maskz_fnmadd_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_fnmsub_round_ph(__m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_fnmsub_round_ph
// CHECK: fneg
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
  return _mm256_fnmsub_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_fnmsub_round_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_maskz_fnmsub_round_ph
// CHECK: fneg
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> zeroinitializer
  return _mm256_maskz_fnmsub_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_fmadd_round_ps(__m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_fmadd_round_ps
// CHECK: @llvm.x86.avx10.vfmaddps256
  return _mm256_fmadd_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_fmadd_round_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_mask_fmadd_round_ps
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmadd_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask3_fmadd_round_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmadd_round_ps
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fmadd_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_fmadd_round_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_maskz_fmadd_round_ps
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> zeroinitializer
  return _mm256_maskz_fmadd_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_fmsub_round_ps(__m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_fmsub_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
  return _mm256_fmsub_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_fmsub_round_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_mask_fmsub_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmsub_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_fmsub_round_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_maskz_fmsub_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> zeroinitializer
  return _mm256_maskz_fmsub_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_fnmadd_round_ps(__m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_fnmadd_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
  return _mm256_fnmadd_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask3_fnmadd_round_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fnmadd_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fnmadd_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_fnmadd_round_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_maskz_fnmadd_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> zeroinitializer
  return _mm256_maskz_fnmadd_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_fnmsub_round_ps(__m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_fnmsub_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
  return _mm256_fnmsub_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_fnmsub_round_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_maskz_fnmsub_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> zeroinitializer
  return _mm256_maskz_fnmsub_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_fmadd_round_pch(__m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_fmadd_round_pch
// CHECK: @llvm.x86.avx10.mask.vfmaddcph256
  return _mm256_fmadd_round_pch(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_fmadd_round_pch(__m256h __A, __mmask8 __U, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_fmadd_round_pch
// CHECK: @llvm.x86.avx10.mask.vfmaddcph256
// CHECK:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmadd_round_pch(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask3_fmadd_round_pch(__m256h __A, __m256h __B, __m256h __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmadd_round_pch
// CHECK: @llvm.x86.avx10.mask.vfmaddcph256
// CHECK-NOT:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fmadd_round_pch(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_fmadd_round_pch(__mmask8 __U, __m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_maskz_fmadd_round_pch
// CHECK: @llvm.x86.avx10.maskz.vfmaddcph256
  return _mm256_maskz_fmadd_round_pch(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_fmaddsub_round_pd(__m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_fmaddsub_round_pd
// CHECK: @llvm.x86.avx10.vfmaddsubpd256
  return _mm256_fmaddsub_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_fmaddsub_round_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_mask_fmaddsub_round_pd
// CHECK: @llvm.x86.avx10.vfmaddsubpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fmaddsub_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask3_fmaddsub_round_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmaddsub_round_pd
// CHECK: @llvm.x86.avx10.vfmaddsubpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fmaddsub_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_fmaddsub_round_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_maskz_fmaddsub_round_pd
// CHECK: @llvm.x86.avx10.vfmaddsubpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> zeroinitializer
  return _mm256_maskz_fmaddsub_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_fmsubadd_round_pd(__m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_fmsubadd_round_pd
// CHECK: fneg <4 x double> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddsubpd256
  return _mm256_fmsubadd_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_fmsubadd_round_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_mask_fmsubadd_round_pd
// CHECK: fneg <4 x double> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddsubpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fmsubadd_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_fmsubadd_round_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_maskz_fmsubadd_round_pd
// CHECK: fneg <4 x double> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddsubpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> zeroinitializer
  return _mm256_maskz_fmsubadd_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_fmaddsub_round_ph(__m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_fmaddsub_round_ph
// CHECK: @llvm.x86.avx10.vfmaddsubph256
  return _mm256_fmaddsub_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_fmaddsub_round_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_fmaddsub_round_ph
// CHECK: @llvm.x86.avx10.vfmaddsubph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fmaddsub_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask3_fmaddsub_round_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmaddsub_round_ph
// CHECK: @llvm.x86.avx10.vfmaddsubph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fmaddsub_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_fmaddsub_round_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_maskz_fmaddsub_round_ph
// CHECK: @llvm.x86.avx10.vfmaddsubph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> zeroinitializer
  return _mm256_maskz_fmaddsub_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_fmsubadd_round_ph(__m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_fmsubadd_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddsubph256
  return _mm256_fmsubadd_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_fmsubadd_round_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_fmsubadd_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddsubph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fmsubadd_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_fmsubadd_round_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_maskz_fmsubadd_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddsubph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> zeroinitializer
  return _mm256_maskz_fmsubadd_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_fmaddsub_round_ps(__m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_fmaddsub_round_ps
// CHECK: @llvm.x86.avx10.vfmaddsubps256
  return _mm256_fmaddsub_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_fmaddsub_round_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_mask_fmaddsub_round_ps
// CHECK: @llvm.x86.avx10.vfmaddsubps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmaddsub_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask3_fmaddsub_round_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmaddsub_round_ps
// CHECK: @llvm.x86.avx10.vfmaddsubps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fmaddsub_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_fmaddsub_round_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_maskz_fmaddsub_round_ps
// CHECK: @llvm.x86.avx10.vfmaddsubps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> zeroinitializer
  return _mm256_maskz_fmaddsub_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_fmsubadd_round_ps(__m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_fmsubadd_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddsubps256
  return _mm256_fmsubadd_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_fmsubadd_round_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_mask_fmsubadd_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddsubps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmsubadd_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_fmsubadd_round_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_maskz_fmsubadd_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddsubps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> zeroinitializer
  return _mm256_maskz_fmsubadd_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask3_fmsub_round_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmsub_round_pd
// CHECK: fneg <4 x double> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fmsub_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask3_fmsubadd_round_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmsubadd_round_pd
// CHECK: fneg <4 x double> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddsubpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fmsubadd_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_fnmadd_round_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_mask_fnmadd_round_pd
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fnmadd_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_fnmsub_round_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
// CHECK-LABEL: @test_mm256_mask_fnmsub_round_pd
// CHECK: fneg <4 x double>
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fnmsub_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask3_fnmsub_round_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fnmsub_round_pd
// CHECK: fneg <4 x double>
// CHECK: fneg <4 x double>
// CHECK: @llvm.x86.avx10.vfmaddpd256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fnmsub_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask3_fmsub_round_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmsub_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fmsub_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask3_fmsubadd_round_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmsubadd_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddsubph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fmsubadd_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_fnmadd_round_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_fnmadd_round_ph
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fnmadd_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_fnmsub_round_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_fnmsub_round_ph
// CHECK: fneg
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fnmsub_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask3_fnmsub_round_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
// CHECK-LABEL: @test_mm256_mask3_fnmsub_round_ph
// CHECK: fneg
// CHECK: fneg
// CHECK: @llvm.x86.avx10.vfmaddph256
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fnmsub_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask3_fmsub_round_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmsub_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fmsub_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask3_fmsubadd_round_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fmsubadd_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddsubps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fmsubadd_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_fnmadd_round_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_mask_fnmadd_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fnmadd_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_fnmsub_round_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
// CHECK-LABEL: @test_mm256_mask_fnmsub_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fnmsub_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask3_fnmsub_round_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_mask3_fnmsub_round_ps
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: fneg <8 x float> %{{.*}}
// CHECK: @llvm.x86.avx10.vfmaddps256
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fnmsub_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mul_round_pch(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mul_round_pch
// CHECK: @llvm.x86.avx10.mask.vfmulcph256
  return _mm256_mul_round_pch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_mul_round_pch(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_mul_round_pch
// CHECK: @llvm.x86.avx10.mask.vfmulcph256
  return _mm256_mask_mul_round_pch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_mul_round_pch(__mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_mul_round_pch
// CHECK: @llvm.x86.avx10.mask.vfmulcph256
  return _mm256_maskz_mul_round_pch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_getexp_round_pd(__m256d __A) {
// CHECK-LABEL: @test_mm256_getexp_round_pd
// CHECK: @llvm.x86.avx10.mask.vgetexppd256
  return _mm256_getexp_round_pd(__A, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_getexp_round_pd(__m256d __W, __mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_mask_getexp_round_pd
// CHECK: @llvm.x86.avx10.mask.vgetexppd256
  return _mm256_mask_getexp_round_pd(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_getexp_round_pd(__mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_maskz_getexp_round_pd
// CHECK: @llvm.x86.avx10.mask.vgetexppd256
  return _mm256_maskz_getexp_round_pd(__U, __A, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_getexp_round_ph(__m256h __A) {
// CHECK-LABEL: @test_mm256_getexp_round_ph
// CHECK: @llvm.x86.avx10.mask.vgetexpph256
  return _mm256_getexp_round_ph(__A, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_getexp_round_ph(__m256h __W, __mmask16 __U, __m256h __A) {
// CHECK-LABEL: @test_mm256_mask_getexp_round_ph
// CHECK: @llvm.x86.avx10.mask.vgetexpph256
  return _mm256_mask_getexp_round_ph(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_getexp_round_ph(__mmask16 __U, __m256h __A) {
// CHECK-LABEL: @test_mm256_maskz_getexp_round_ph
// CHECK: @llvm.x86.avx10.mask.vgetexpph256
  return _mm256_maskz_getexp_round_ph(__U, __A, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_getexp_round_ps(__m256 __A) {
// CHECK-LABEL: @test_mm256_getexp_round_ps
// CHECK: @llvm.x86.avx10.mask.vgetexpps256
  return _mm256_getexp_round_ps(__A, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_getexp_round_ps(__m256 __W, __mmask8 __U, __m256 __A) {
// CHECK-LABEL: @test_mm256_mask_getexp_round_ps
// CHECK: @llvm.x86.avx10.mask.vgetexpps256
  return _mm256_mask_getexp_round_ps(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_getexp_round_ps(__mmask8 __U, __m256 __A) {
// CHECK-LABEL: @test_mm256_maskz_getexp_round_ps
// CHECK: @llvm.x86.avx10.mask.vgetexpps256
  return _mm256_maskz_getexp_round_ps(__U, __A, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_getmant_round_pd(__m256d __A) {
// CHECK-LABEL: @test_mm256_getmant_round_pd
// CHECK: @llvm.x86.avx10.mask.vgetmantpd256
  return _mm256_getmant_round_pd(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_getmant_round_pd(__m256d __W, __mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_mask_getmant_round_pd
// CHECK: @llvm.x86.avx10.mask.vgetmantpd256
  return _mm256_mask_getmant_round_pd(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_getmant_round_pd(__mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_maskz_getmant_round_pd
// CHECK: @llvm.x86.avx10.mask.vgetmantpd256
  return _mm256_maskz_getmant_round_pd(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_getmant_round_ph(__m256h __A) {
// CHECK-LABEL: @test_mm256_getmant_round_ph
// CHECK: @llvm.x86.avx10.mask.vgetmantph256
  return _mm256_getmant_round_ph(__A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_getmant_round_ph(__m256h __W, __mmask16 __U, __m256h __A) {
// CHECK-LABEL: @test_mm256_mask_getmant_round_ph
// CHECK: @llvm.x86.avx10.mask.vgetmantph256
  return _mm256_mask_getmant_round_ph(__W, __U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_getmant_round_ph(__mmask16 __U, __m256h __A) {
// CHECK-LABEL: @test_mm256_maskz_getmant_round_ph
// CHECK: @llvm.x86.avx10.mask.vgetmantph256
  return _mm256_maskz_getmant_round_ph(__U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_getmant_round_ps(__m256 __A) {
// CHECK-LABEL: @test_mm256_getmant_round_ps
// CHECK: @llvm.x86.avx10.mask.vgetmantps256
  return _mm256_getmant_round_ps(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_getmant_round_ps(__m256 __W, __mmask8 __U, __m256 __A) {
// CHECK-LABEL: @test_mm256_mask_getmant_round_ps
// CHECK: @llvm.x86.avx10.mask.vgetmantps256
  return _mm256_mask_getmant_round_ps(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_getmant_round_ps(__mmask8 __U, __m256 __A) {
// CHECK-LABEL: @test_mm256_maskz_getmant_round_ps
// CHECK: @llvm.x86.avx10.mask.vgetmantps256
  return _mm256_maskz_getmant_round_ps(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_max_round_pd(__m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_max_round_pd
// CHECK: @llvm.x86.avx10.vmaxpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 8)
  return _mm256_max_round_pd(__A, __B, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_max_round_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mask_max_round_pd
// CHECK: @llvm.x86.avx10.vmaxpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 8)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_max_round_pd(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_max_round_pd(__mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_maskz_max_round_pd
// CHECK: @llvm.x86.avx10.vmaxpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 8)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_max_round_pd(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_max_round_ph(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_max_round_ph
// CHECK: @llvm.x86.avx10.vmaxph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 8)
  return _mm256_max_round_ph(__A, __B, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_max_round_ph(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_max_round_ph
// CHECK: @llvm.x86.avx10.vmaxph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 8)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_max_round_ph(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_max_round_ph(__mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_max_round_ph
// CHECK: @llvm.x86.avx10.vmaxph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 8)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_max_round_ph(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_max_round_ps(__m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_max_round_ps
// CHECK: @llvm.x86.avx10.vmaxps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 8)
  return _mm256_max_round_ps(__A, __B, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_max_round_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mask_max_round_ps
// CHECK: @llvm.x86.avx10.vmaxps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 8)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_max_round_ps(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_max_round_ps(__mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_maskz_max_round_ps
// CHECK: @llvm.x86.avx10.vmaxps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 8)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_max_round_ps(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_min_round_pd(__m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_min_round_pd
// CHECK: @llvm.x86.avx10.vminpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 8)
  return _mm256_min_round_pd(__A, __B, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_min_round_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mask_min_round_pd
// CHECK: @llvm.x86.avx10.vminpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 8)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_min_round_pd(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_min_round_pd(__mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_maskz_min_round_pd
// CHECK: @llvm.x86.avx10.vminpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 8)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_min_round_pd(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_min_round_ph(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_min_round_ph
// CHECK: @llvm.x86.avx10.vminph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 8)
  return _mm256_min_round_ph(__A, __B, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_min_round_ph(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_min_round_ph
// CHECK: @llvm.x86.avx10.vminph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 8)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_min_round_ph(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_min_round_ph(__mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_min_round_ph
// CHECK: @llvm.x86.avx10.vminph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 8)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_min_round_ph(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_min_round_ps(__m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_min_round_ps
// CHECK: @llvm.x86.avx10.vminps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 8)
  return _mm256_min_round_ps(__A, __B, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_min_round_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mask_min_round_ps
// CHECK: @llvm.x86.avx10.vminps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 8)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_min_round_ps(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_min_round_ps(__mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_maskz_min_round_ps
// CHECK: @llvm.x86.avx10.vminps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 8)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_min_round_ps(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mul_round_pd(__m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mul_round_pd
// CHECK: @llvm.x86.avx10.vmulpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 11)
  return _mm256_mul_round_pd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_mul_round_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mask_mul_round_pd
// CHECK: @llvm.x86.avx10.vmulpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 10)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_mul_round_pd(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_mul_round_pd(__mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_maskz_mul_round_pd
// CHECK: @llvm.x86.avx10.vmulpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 9)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_mul_round_pd(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mul_round_ph(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mul_round_ph
// CHECK: @llvm.x86.avx10.vmulph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 11)
  return _mm256_mul_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_mul_round_ph(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_mul_round_ph
// CHECK: @llvm.x86.avx10.vmulph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 10)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_mul_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_mul_round_ph(__mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_mul_round_ph
// CHECK: @llvm.x86.avx10.vmulph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 9)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_mul_round_ph(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mul_round_ps(__m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mul_round_ps
// CHECK: @llvm.x86.avx10.vmulps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 11)
  return _mm256_mul_round_ps(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_mul_round_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mask_mul_round_ps
// CHECK: @llvm.x86.avx10.vmulps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 10)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_mul_round_ps(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_mul_round_ps(__mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_maskz_mul_round_ps
// CHECK: @llvm.x86.avx10.vmulps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 9)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_mul_round_ps(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_range_round_pd(__m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_range_round_pd
// CHECK: @llvm.x86.avx10.mask.vrangepd256
  return _mm256_range_round_pd(__A, __B, 4, 8);
}

__m256d test_mm256_mask_range_round_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mask_range_round_pd
// CHECK: @llvm.x86.avx10.mask.vrangepd256
  return _mm256_mask_range_round_pd(__W, __U, __A, __B, 4, 8);
}

__m256d test_mm256_maskz_range_round_pd(__mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_maskz_range_round_pd
// CHECK: @llvm.x86.avx10.mask.vrangepd256
  return _mm256_maskz_range_round_pd(__U, __A, __B, 4, 8);
}

__m256 test_mm256_range_round_ps(__m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_range_round_ps
// CHECK: @llvm.x86.avx10.mask.vrangeps256
  return _mm256_range_round_ps(__A, __B, 4, 8);
}

__m256 test_mm256_mask_range_round_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mask_range_round_ps
// CHECK: @llvm.x86.avx10.mask.vrangeps256
  return _mm256_mask_range_round_ps(__W, __U, __A, __B, 4, 8);
}

__m256 test_mm256_maskz_range_round_ps(__mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_maskz_range_round_ps
// CHECK: @llvm.x86.avx10.mask.vrangeps256
  return _mm256_maskz_range_round_ps(__U, __A, __B, 4, 8);
}

__m256d test_mm256_reduce_round_pd(__m256d __A) {
// CHECK-LABEL: @test_mm256_reduce_round_pd
// CHECK: @llvm.x86.avx10.mask.vreducepd256
  return _mm256_reduce_round_pd(__A, 4, 8);
}

__m256d test_mm256_mask_reduce_round_pd(__m256d __W, __mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_mask_reduce_round_pd
// CHECK: @llvm.x86.avx10.mask.vreducepd256
  return _mm256_mask_reduce_round_pd(__W, __U, __A, 4, 8);
}

__m256d test_mm256_maskz_reduce_round_pd(__mmask8 __U, __m256d __A) {
// CHECK-LABEL: @test_mm256_maskz_reduce_round_pd
// CHECK: @llvm.x86.avx10.mask.vreducepd256
  return _mm256_maskz_reduce_round_pd(__U, __A, 4, 8);
}

__m256h test_mm256_mask_reduce_round_ph(__m256h __A, __mmask8 __U, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_reduce_round_ph
// CHECK: @llvm.x86.avx10.mask.vreduceph256
  return _mm256_mask_reduce_round_ph(__A, __U, __C, 3, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_reduce_round_ph(__m256h __A, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_maskz_reduce_round_ph
// CHECK: @llvm.x86.avx10.mask.vreduceph256
  return _mm256_maskz_reduce_round_ph(__U, __A, 3, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_reduce_round_ph(__m256h __A) {
// CHECK-LABEL: @test_mm256_reduce_round_ph
// CHECK: @llvm.x86.avx10.mask.vreduceph256
  return _mm256_reduce_round_ph(__A, 3, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_reduce_round_ps(__m256 __A) {
// CHECK-LABEL: @test_mm256_reduce_round_ps
// CHECK: @llvm.x86.avx10.mask.vreduceps256
  return _mm256_reduce_round_ps(__A, 4, 8);
}

__m256 test_mm256_mask_reduce_round_ps(__m256 __W, __mmask8 __U, __m256 __A) {
// CHECK-LABEL: @test_mm256_mask_reduce_round_ps
// CHECK: @llvm.x86.avx10.mask.vreduceps256
  return _mm256_mask_reduce_round_ps(__W, __U, __A, 4, 8);
}

__m256 test_mm256_maskz_reduce_round_ps(__mmask8 __U, __m256 __A) {
// CHECK-LABEL: @test_mm256_maskz_reduce_round_ps
// CHECK: @llvm.x86.avx10.mask.vreduceps256
  return _mm256_maskz_reduce_round_ps(__U, __A, 4, 8);
}

__m256d test_mm256_roundscale_round_pd(__m256d __A)
{
// CHECK-LABEL: @test_mm256_roundscale_round_pd
// CHECK: @llvm.x86.avx10.mask.vrndscalepd256
  return _mm256_roundscale_round_pd(__A,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_roundscale_round_pd(__m256d __A,__mmask8 __U,__m256d __C)
{
// CHECK-LABEL: @test_mm256_mask_roundscale_round_pd
// CHECK: @llvm.x86.avx10.mask.vrndscalepd256
  return _mm256_mask_roundscale_round_pd(__A,__U,__C,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_roundscale_round_pd(__m256d __A,__mmask8 __U)
{
// CHECK-LABEL: @test_mm256_maskz_roundscale_round_pd
// CHECK: @llvm.x86.avx10.mask.vrndscalepd256
  return _mm256_maskz_roundscale_round_pd(__U,__A,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_roundscale_round_ph(__m256h __A, __mmask8 __U, __m256h __C) {
// CHECK-LABEL: @test_mm256_mask_roundscale_round_ph
// CHECK: @llvm.x86.avx10.mask.vrndscaleph256
  return _mm256_mask_roundscale_round_ph(__A, __U, __C, 3, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_roundscale_round_ph(__m256h __A, __mmask8 __U) {
// CHECK-LABEL: @test_mm256_maskz_roundscale_round_ph
// CHECK: @llvm.x86.avx10.mask.vrndscaleph256
  return _mm256_maskz_roundscale_round_ph(__U, __A, 3, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_roundscale_round_ph(__m256h __A) {
// CHECK-LABEL: @test_mm256_roundscale_round_ph
// CHECK: @llvm.x86.avx10.mask.vrndscaleph256
  return _mm256_roundscale_round_ph(__A, 3, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_roundscale_round_ps(__m256 __A)
{
// CHECK-LABEL: @test_mm256_roundscale_round_ps
// CHECK: @llvm.x86.avx10.mask.vrndscaleps256
  return _mm256_roundscale_round_ps(__A,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_roundscale_round_ps(__m256 __A,__mmask8 __U,__m256 __C)
{
// CHECK-LABEL: @test_mm256_mask_roundscale_round_ps
// CHECK: @llvm.x86.avx10.mask.vrndscaleps256
  return _mm256_mask_roundscale_round_ps(__A,__U,__C,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_roundscale_round_ps(__m256 __A,__mmask8 __U)
{
// CHECK-LABEL: @test_mm256_maskz_roundscale_round_ps
// CHECK: @llvm.x86.avx10.mask.vrndscaleps256
  return _mm256_maskz_roundscale_round_ps(__U,__A,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m256d test_mm256_scalef_round_pd(__m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_scalef_round_pd
// CHECK: @llvm.x86.avx10.mask.vscalefpd256
  return _mm256_scalef_round_pd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_scalef_round_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mask_scalef_round_pd
// CHECK: @llvm.x86.avx10.mask.vscalefpd256
  return _mm256_mask_scalef_round_pd(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_scalef_round_pd(__mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_maskz_scalef_round_pd
// CHECK: @llvm.x86.avx10.mask.vscalefpd256
  return _mm256_maskz_scalef_round_pd(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_scalef_round_ph(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_scalef_round_ph
// CHECK: @llvm.x86.avx10.mask.vscalefph256
  return _mm256_scalef_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_scalef_round_ph(__m256h __W, __mmask16 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_scalef_round_ph
// CHECK: @llvm.x86.avx10.mask.vscalefph256
  return _mm256_mask_scalef_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_scalef_round_ph(__mmask16 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_scalef_round_ph
// CHECK: @llvm.x86.avx10.mask.vscalefph256
  return _mm256_maskz_scalef_round_ph(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_scalef_round_ps(__m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_scalef_round_ps
// CHECK: @llvm.x86.avx10.mask.vscalefps256
  return _mm256_scalef_round_ps(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_scalef_round_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mask_scalef_round_ps
// CHECK: @llvm.x86.avx10.mask.vscalefps256
  return _mm256_mask_scalef_round_ps(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_scalef_round_ps(__mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_maskz_scalef_round_ps
// CHECK: @llvm.x86.avx10.mask.vscalefps256
  return _mm256_maskz_scalef_round_ps(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_sqrt_round_pd(__m256d __A)
{
// CHECK-LABEL: @test_mm256_sqrt_round_pd
// CHECK: call <4 x double> @llvm.x86.avx10.vsqrtpd256(<4 x double> %{{.*}}, i32 11)
  return _mm256_sqrt_round_pd(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_sqrt_round_pd(__m256d __W,__mmask8 __U,__m256d __A)
{
// CHECK-LABEL: @test_mm256_mask_sqrt_round_pd
// CHECK: call <4 x double> @llvm.x86.avx10.vsqrtpd256(<4 x double> %{{.*}}, i32 11)
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_sqrt_round_pd(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_sqrt_round_pd(__mmask8 __U,__m256d __A)
{
// CHECK-LABEL: @test_mm256_maskz_sqrt_round_pd
// CHECK: call <4 x double> @llvm.x86.avx10.vsqrtpd256(<4 x double> %{{.*}}, i32 11)
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> {{.*}}
  return _mm256_maskz_sqrt_round_pd(__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_sqrt_round_ph(__m256h __A) {
// CHECK-LABEL: @test_mm256_sqrt_round_ph
// CHECK: call <16 x half> @llvm.x86.avx10.vsqrtph256(<16 x half> %{{.*}}, i32 11)
  return _mm256_sqrt_round_ph(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_sqrt_round_ph(__m256h __W, __mmask16 __U, __m256h __A) {
// CHECK-LABEL: @test_mm256_mask_sqrt_round_ph
// CHECK: call <16 x half> @llvm.x86.avx10.vsqrtph256(<16 x half> %{{.*}}, i32 11)
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_sqrt_round_ph(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_sqrt_round_ph(__mmask16 __U, __m256h __A) {
// CHECK-LABEL: @test_mm256_maskz_sqrt_round_ph
// CHECK: call <16 x half> @llvm.x86.avx10.vsqrtph256(<16 x half> %{{.*}}, i32 11)
// CHECK: bitcast i16 %{{.*}} to <16 x i1>
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> {{.*}}
  return _mm256_maskz_sqrt_round_ph(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_sqrt_round_ps(__m256 __A)
{
// CHECK-LABEL: @test_mm256_sqrt_round_ps
// CHECK: call <8 x float> @llvm.x86.avx10.vsqrtps256(<8 x float> %{{.*}}, i32 11)
  return _mm256_sqrt_round_ps(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_sqrt_round_ps(__m256 __W,__mmask8 __U,__m256 __A)
{
// CHECK-LABEL: @test_mm256_mask_sqrt_round_ps
// CHECK: call <8 x float> @llvm.x86.avx10.vsqrtps256(<8 x float> %{{.*}}, i32 11)
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_sqrt_round_ps(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_sqrt_round_ps(__mmask8 __U,__m256 __A)
{
// CHECK-LABEL: @test_mm256_maskz_sqrt_round_ps
// CHECK: call <8 x float> @llvm.x86.avx10.vsqrtps256(<8 x float> %{{.*}}, i32 11)
// CHECK: bitcast i8 %{{.*}} to <8 x i1>
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> {{.*}}
  return _mm256_maskz_sqrt_round_ps(__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_sub_round_pd(__m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_sub_round_pd
// CHECK: @llvm.x86.avx10.vsubpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 11)
  return _mm256_sub_round_pd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_sub_round_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mask_sub_round_pd
// CHECK: @llvm.x86.avx10.vsubpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 10)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_sub_round_pd(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_sub_round_pd(__mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_maskz_sub_round_pd
// CHECK: @llvm.x86.avx10.vsubpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 9)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_sub_round_pd(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_sub_round_ph(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_sub_round_ph
// CHECK: @llvm.x86.avx10.vsubph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 11)
  return _mm256_sub_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_sub_round_ph(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_sub_round_ph
// CHECK: @llvm.x86.avx10.vsubph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 10)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_sub_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_sub_round_ph(__mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_sub_round_ph
// CHECK: @llvm.x86.avx10.vsubph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 9)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_sub_round_ph(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_sub_round_ps(__m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_sub_round_ps
// CHECK: @llvm.x86.avx10.vsubps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 11)
  return _mm256_sub_round_ps(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_sub_round_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mask_sub_round_ps
// CHECK: @llvm.x86.avx10.vsubps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 10)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_sub_round_ps(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_sub_round_ps(__mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_maskz_sub_round_ps
// CHECK: @llvm.x86.avx10.vsubps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 9)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_sub_round_ps(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}
