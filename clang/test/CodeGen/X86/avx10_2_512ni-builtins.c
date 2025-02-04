// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64 -target-feature +avx10.2-512 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i686 -target-feature +avx10.2-512 -emit-llvm -o - | FileCheck %s

#include <immintrin.h>

// VNNI FP16
__m512 test_mm512_dpph_ps(__m512 __W, __m512h __A, __m512h __B) {
// CHECK-LABEL: @test_mm512_dpph_ps(
// CHECK: call <16 x float> @llvm.x86.avx10.vdpphps.512
  return _mm512_dpph_ps(__W, __A, __B);
}

__m512 test_mm512_mask_dpph_ps(__m512 __W, __mmask16 __U, __m512h __A, __m512h __B) {
// CHECK-LABEL: @test_mm512_mask_dpph_ps(
// CHECK: call <16 x float> @llvm.x86.avx10.vdpphps.512
// CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_dpph_ps(__W, __U, __A, __B);
}

__m512 test_mm512_maskz_dpph_ps(__mmask16 __U, __m512 __W, __m512h __A, __m512h __B) {
// CHECK-LABEL: @test_mm512_maskz_dpph_ps(
// CHECK: call <16 x float> @llvm.x86.avx10.vdpphps.512
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_dpph_ps(__U, __W, __A, __B);
}

// VMPSADBW
__m512i test_mm512_mpsadbw_epu8(__m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_mpsadbw_epu8
// CHECK: @llvm.x86.avx10.vmpsadbw.512
  return _mm512_mpsadbw_epu8(__A, __B, 17);
}

__m512i test_mm512_mask_mpsadbw_epu8(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_mask_mpsadbw_epu8
// CHECK: @llvm.x86.avx10.vmpsadbw.512
// CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_mpsadbw_epu8(__W, __U, __A, __B, 17);
}

__m512i test_mm512_maskz_mpsadbw_epu8(__mmask32 __U, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_maskz_mpsadbw_epu8
// CHECK: @llvm.x86.avx10.vmpsadbw.512
// CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_mpsadbw_epu8(__U, __A, __B, 17);
}

// VNNI INT8
__m512i test_mm512_dpbssd_epi32(__m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_dpbssd_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbssd.512
  return _mm512_dpbssd_epi32(__W, __A, __B);
}

__m512i test_mm512_mask_dpbssd_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_mask_dpbssd_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbssd.512
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpbssd_epi32(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_dpbssd_epi32(__mmask16 __U, __m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_maskz_dpbssd_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbssd.512
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpbssd_epi32(__U, __W, __A, __B);
}

__m512i test_mm512_dpbssds_epi32(__m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_dpbssds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbssds.512
  return _mm512_dpbssds_epi32(__W, __A, __B);
}

__m512i test_mm512_mask_dpbssds_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_mask_dpbssds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbssds.512
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpbssds_epi32(__W, __U,  __A, __B);
}

__m512i test_mm512_maskz_dpbssds_epi32(__mmask16 __U, __m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_maskz_dpbssds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbssds.512
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpbssds_epi32(__U, __W, __A, __B);
}

__m512i test_mm512_dpbsud_epi32(__m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_dpbsud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbsud.512
  return _mm512_dpbsud_epi32(__W, __A, __B);
}

__m512i test_mm512_mask_dpbsud_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_mask_dpbsud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbsud.512
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpbsud_epi32(__W, __U,  __A, __B);
}

__m512i test_mm512_maskz_dpbsud_epi32(__mmask16 __U, __m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_maskz_dpbsud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbsud.512
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpbsud_epi32(__U, __W, __A, __B);
}

__m512i test_mm512_dpbsuds_epi32(__m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_dpbsuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbsuds.512
  return _mm512_dpbsuds_epi32(__W, __A, __B);
}

__m512i test_mm512_mask_dpbsuds_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_mask_dpbsuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbsuds.512
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpbsuds_epi32(__W, __U,  __A, __B);
}

__m512i test_mm512_maskz_dpbsuds_epi32(__mmask16 __U, __m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_maskz_dpbsuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbsuds.512
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpbsuds_epi32(__U, __W, __A, __B);
}

__m512i test_mm512_dpbuud_epi32(__m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_dpbuud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbuud.512
  return _mm512_dpbuud_epi32(__W, __A, __B);
}

__m512i test_mm512_mask_dpbuud_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_mask_dpbuud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbuud.512
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpbuud_epi32(__W, __U,  __A, __B);
}

__m512i test_mm512_maskz_dpbuud_epi32(__mmask16 __U, __m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_maskz_dpbuud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbuud.512
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpbuud_epi32(__U, __W, __A, __B);
}

__m512i test_mm512_dpbuuds_epi32(__m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_dpbuuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512
  return _mm512_dpbuuds_epi32(__W, __A, __B);
}

__m512i test_mm512_mask_dpbuuds_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_mask_dpbuuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpbuuds_epi32(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_dpbuuds_epi32(__mmask16 __U, __m512i __W, __m512i __A, __m512i __B) {
// CHECK-LABEL: @test_mm512_maskz_dpbuuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpbuuds_epi32(__U, __W, __A, __B);
}

/* VNNI INT16 */
__m512i test_mm512_dpwsud_epi32(__m512i __A, __m512i __B, __m512i __C) {
// CHECK-LABEL: @test_mm512_dpwsud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwsud.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_dpwsud_epi32(__A, __B, __C);
}

__m512i test_mm512_mask_dpwsud_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_mask_dpwsud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwsud.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpwsud_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_maskz_dpwsud_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_maskz_dpwsud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwsud.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpwsud_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_dpwsuds_epi32(__m512i __A, __m512i __B, __m512i __C) {
// CHECK-LABEL: @test_mm512_dpwsuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwsuds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_dpwsuds_epi32(__A, __B, __C);
}

__m512i test_mm512_mask_dpwsuds_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_mask_dpwsuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwsuds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpwsuds_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_maskz_dpwsuds_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_maskz_dpwsuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwsuds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpwsuds_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_dpwusd_epi32(__m512i __A, __m512i __B, __m512i __C) {
// CHECK-LABEL: @test_mm512_dpwusd_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_dpwusd_epi32(__A, __B, __C);
}

__m512i test_mm512_mask_dpwusd_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_mask_dpwusd_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpwusd_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_maskz_dpwusd_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_maskz_dpwusd_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpwusd_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_dpwusds_epi32(__m512i __A, __m512i __B, __m512i __C) {
// CHECK-LABEL: @test_mm512_dpwusds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_dpwusds_epi32(__A, __B, __C);
}

__m512i test_mm512_mask_dpwusds_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_mask_dpwusds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpwusds_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_maskz_dpwusds_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_maskz_dpwusds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpwusds_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_dpwuud_epi32(__m512i __A, __m512i __B, __m512i __C) {
// CHECK-LABEL: @test_mm512_dpwuud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_dpwuud_epi32(__A, __B, __C);
}

__m512i test_mm512_mask_dpwuud_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_mask_dpwuud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpwuud_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_maskz_dpwuud_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_maskz_dpwuud_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpwuud_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_dpwuuds_epi32(__m512i __A, __m512i __B, __m512i __C) {
// CHECK-LABEL: @test_mm512_dpwuuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_dpwuuds_epi32(__A, __B, __C);
}

__m512i test_mm512_mask_dpwuuds_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_mask_dpwuuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpwuuds_epi32(__A, __B, __C, __D);
}

__m512i test_mm512_maskz_dpwuuds_epi32(__m512i __A, __mmask16 __B, __m512i __C, __m512i __D) {
// CHECK-LABEL: @test_mm512_maskz_dpwuuds_epi32(
// CHECK: call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: zeroinitializer
// CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpwuuds_epi32(__A, __B, __C, __D);
}
