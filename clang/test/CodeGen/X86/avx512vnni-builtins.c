//  RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vnni -emit-llvm -o - -Wall -Werror | FileCheck %s
//  RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vnni -emit-llvm -o - -Wall -Werror | FileCheck %s
//  RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vnni -emit-llvm -o - -Wall -Werror | FileCheck %s
//  RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vnni -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512i test_mm512_mask_dpbusd_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_dpbusd_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpbusd_epi32(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_dpbusd_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_dpbusd_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpbusd_epi32(__U, __S, __A, __B);
}

__m512i test_mm512_dpbusd_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_dpbusd_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_dpbusd_epi32(__S, __A, __B);
}

__m512i test_mm512_mask_dpbusds_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_dpbusds_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpbusds_epi32(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_dpbusds_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_dpbusds_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpbusds_epi32(__U, __S, __A, __B);
}

__m512i test_mm512_dpbusds_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_dpbusds_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_dpbusds_epi32(__S, __A, __B);
}

__m512i test_mm512_mask_dpwssd_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_dpwssd_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpwssd_epi32(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_dpwssd_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_dpwssd_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpwssd_epi32(__U, __S, __A, __B);
}

__m512i test_mm512_dpwssd_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_dpwssd_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_dpwssd_epi32(__S, __A, __B);
}

__m512i test_mm512_mask_dpwssds_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_dpwssds_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_dpwssds_epi32(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_dpwssds_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_dpwssds_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_dpwssds_epi32(__U, __S, __A, __B);
}

__m512i test_mm512_dpwssds_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_dpwssds_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_dpwssds_epi32(__S, __A, __B);
}

