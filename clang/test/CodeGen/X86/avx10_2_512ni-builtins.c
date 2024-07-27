// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +avx10.2-512 -emit-llvm -o - | FileCheck %s

#include <immintrin.h>

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
