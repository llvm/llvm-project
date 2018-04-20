// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <x86intrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/avx2-intrinsics-fast-isel.ll

__m256i test_mm256_abs_epi8(__m256i a) {
  // CHECK-LABEL: test_mm256_abs_epi8
  // CHECK: [[SUB:%.*]] = sub <32 x i8> zeroinitializer, %{{.*}}
  // CHECK: [[CMP:%.*]] = icmp sgt <32 x i8> %{{.*}}, zeroinitializer
  // CHECK: select <32 x i1> [[CMP]], <32 x i8> %{{.*}}, <32 x i8> [[SUB]]
  return _mm256_abs_epi8(a);
}

__m256i test_mm256_abs_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_abs_epi16
  // CHECK: [[SUB:%.*]] = sub <16 x i16> zeroinitializer, %{{.*}}
  // CHECK: [[CMP:%.*]] = icmp sgt <16 x i16> %{{.*}}, zeroinitializer
  // CHECK: select <16 x i1> [[CMP]], <16 x i16> %{{.*}}, <16 x i16> [[SUB]]
  return _mm256_abs_epi16(a);
}

__m256i test_mm256_abs_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_abs_epi32
  // CHECK: [[SUB:%.*]] = sub <8 x i32> zeroinitializer, %{{.*}}
  // CHECK: [[CMP:%.*]] = icmp sgt <8 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <8 x i1> [[CMP]], <8 x i32> %{{.*}}, <8 x i32> [[SUB]]
  return _mm256_abs_epi32(a);
}

__m256i test_mm256_add_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_add_epi8
  // CHECK: add <32 x i8>
  return _mm256_add_epi8(a, b);
}

__m256i test_mm256_add_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_add_epi16
  // CHECK: add <16 x i16>
  return _mm256_add_epi16(a, b);
}

__m256i test_mm256_add_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_add_epi32
  // CHECK: add <8 x i32>
  return _mm256_add_epi32(a, b);
}

__m256i test_mm256_add_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_add_epi64
  // CHECK: add <4 x i64>
  return _mm256_add_epi64(a, b);
}

__m256i test_mm256_adds_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_adds_epi8
  // CHECK-NOT: call <32 x i8> @llvm.x86.avx2.padds.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: add <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: icmp sle <32 x i16> %{{.*}}, <i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> <i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127>
  // CHECK: icmp slt <32 x i16> %{{.*}}, <i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> <i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128>, <32 x i16> %{{.*}}
  // CHECK: trunc <32 x i16> %{{.*}} to <32 x i8>
  return _mm256_adds_epi8(a, b);
}

__m256i test_mm256_adds_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_adds_epi16
  // CHECK-NOT: call <16 x i16> @llvm.x86.avx2.padds.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: add <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp sle <16 x i32> %{{.*}}, <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  // CHECK: icmp slt <16 x i32> %{{.*}}, <i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> <i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768>, <16 x i32> %{{.*}}
  // CHECK: trunc <16 x i32> %{{.*}} to <16 x i16>
  return _mm256_adds_epi16(a, b);
}

__m256i test_mm256_adds_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_adds_epu8
  // CHECK-NOT: call <32 x i8> @llvm.x86.avx2.paddus.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: add <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: icmp ule <32 x i16> %{{.*}}, <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>
  // CHECK: trunc <32 x i16> %{{.*}} to <32 x i8>
  return _mm256_adds_epu8(a, b);
}

__m256i test_mm256_adds_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_adds_epu16
  // CHECK-NOT: call <16 x i16> @llvm.x86.avx2.paddus.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: add <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp ule <16 x i32> %{{.*}}, <i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535>
  // CHECK: trunc <16 x i32> %{{.*}} to <16 x i16>
  return _mm256_adds_epu16(a, b);
}

__m256i test_mm256_alignr_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  return _mm256_alignr_epi8(a, b, 2);
}

__m256i test2_mm256_alignr_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test2_mm256_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> zeroinitializer, <32 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48>
  return _mm256_alignr_epi8(a, b, 17);
}

__m256i test_mm256_and_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_and_si256
  // CHECK: and <4 x i64>
  return _mm256_and_si256(a, b);
}

__m256i test_mm256_andnot_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_andnot_si256
  // CHECK: xor <4 x i64>
  // CHECK: and <4 x i64>
  return _mm256_andnot_si256(a, b);
}

__m256i test_mm256_avg_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_avg_epu8
  // CHECK-NOT: call <32 x i8> @llvm.x86.avx2.pavg.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: add <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: add <32 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: lshr <32 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: trunc <32 x i16> %{{.*}} to <32 x i8>
  return _mm256_avg_epu8(a, b);
}

__m256i test_mm256_avg_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_avg_epu16
  // CHECK-NOT: call <16 x i16> @llvm.x86.avx2.pavg.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: add <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: add <16 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: lshr <16 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: trunc <16 x i32> %{{.*}} to <16 x i16>
  return _mm256_avg_epu16(a, b);
}

// FIXME: We should also lower the __builtin_ia32_pblendw128 (and similar)
// functions to this IR. In the future we could delete the corresponding
// intrinsic in LLVM if it's not being used anymore.
__m256i test_mm256_blend_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_blend_epi16
  // CHECK-NOT: @llvm.x86.avx2.pblendw
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 17, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 25, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm256_blend_epi16(a, b, 2);
}

__m128i test_mm_blend_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_blend_epi32
  // CHECK-NOT: @llvm.x86.avx2.pblendd.128
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  return _mm_blend_epi32(a, b, 0x35);
}

__m256i test_mm256_blend_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_blend_epi32
  // CHECK-NOT: @llvm.x86.avx2.pblendd.256
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 13, i32 6, i32 7>
  return _mm256_blend_epi32(a, b, 0x35);
}

__m256i test_mm256_blendv_epi8(__m256i a, __m256i b, __m256i m) {
  // CHECK-LABEL: test_mm256_blendv_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_blendv_epi8(a, b, m);
}

__m128i test_mm_broadcastb_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastb_epi8
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastb.128
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> zeroinitializer
  return _mm_broadcastb_epi8(a);
}

__m256i test_mm256_broadcastb_epi8(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastb_epi8
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastb.256
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <32 x i32> zeroinitializer
  return _mm256_broadcastb_epi8(a);
}

__m128i test_mm_broadcastd_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastd_epi32
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastd.128
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> zeroinitializer
  return _mm_broadcastd_epi32(a);
}

__m256i test_mm256_broadcastd_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastd_epi32
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastd.256
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> zeroinitializer
  return _mm256_broadcastd_epi32(a);
}

__m128i test_mm_broadcastq_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastq_epi64
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastq.128
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> zeroinitializer
  return _mm_broadcastq_epi64(a);
}

__m256i test_mm256_broadcastq_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastq_epi64
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastq.256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> zeroinitializer
  return _mm256_broadcastq_epi64(a);
}

__m128d test_mm_broadcastsd_pd(__m128d a) {
  // CHECK-LABEL: test_mm_broadcastsd_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  return _mm_broadcastsd_pd(a);
}

__m256d test_mm256_broadcastsd_pd(__m128d a) {
  // CHECK-LABEL: test_mm256_broadcastsd_pd
  // CHECK-NOT: @llvm.x86.avx2.vbroadcast.sd.pd.256
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> zeroinitializer
  return _mm256_broadcastsd_pd(a);
}

__m256i test_mm256_broadcastsi128_si256(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastsi128_si256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  return _mm256_broadcastsi128_si256(a);
}

__m128 test_mm_broadcastss_ps(__m128 a) {
  // CHECK-LABEL: test_mm_broadcastss_ps
  // CHECK-NOT: @llvm.x86.avx2.vbroadcast.ss.ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> zeroinitializer
  return _mm_broadcastss_ps(a);
}

__m256 test_mm256_broadcastss_ps(__m128 a) {
  // CHECK-LABEL: test_mm256_broadcastss_ps
  // CHECK-NOT: @llvm.x86.avx2.vbroadcast.ss.ps.256
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> zeroinitializer
  return _mm256_broadcastss_ps(a);
}

__m128i test_mm_broadcastw_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastw_epi16
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastw.128
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> zeroinitializer
  return _mm_broadcastw_epi16(a);
}

__m256i test_mm256_broadcastw_epi16(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastw_epi16
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastw.256
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i32> zeroinitializer
  return _mm256_broadcastw_epi16(a);
}

__m256i test_mm256_bslli_epi128(__m256i a) {
  // CHECK-LABEL: test_mm256_bslli_epi128
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60>
  return _mm256_bslli_epi128(a, 3);
}

__m256i test_mm256_bsrli_epi128(__m256i a) {
  // CHECK-LABEL: test_mm256_bsrli_epi128
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50>
  return _mm256_bsrli_epi128(a, 3);
}

__m256i test_mm256_cmpeq_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi8
  // CHECK: icmp eq <32 x i8>
  return _mm256_cmpeq_epi8(a, b);
}

__m256i test_mm256_cmpeq_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi16
  // CHECK: icmp eq <16 x i16>
  return _mm256_cmpeq_epi16(a, b);
}

__m256i test_mm256_cmpeq_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi32
  // CHECK: icmp eq <8 x i32>
  return _mm256_cmpeq_epi32(a, b);
}

__m256i test_mm256_cmpeq_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi64
  // CHECK: icmp eq <4 x i64>
  return _mm256_cmpeq_epi64(a, b);
}

__m256i test_mm256_cmpgt_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi8
  // CHECK: icmp sgt <32 x i8>
  return _mm256_cmpgt_epi8(a, b);
}

__m256i test_mm256_cmpgt_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi16
  // CHECK: icmp sgt <16 x i16>
  return _mm256_cmpgt_epi16(a, b);
}

__m256i test_mm256_cmpgt_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi32
  // CHECK: icmp sgt <8 x i32>
  return _mm256_cmpgt_epi32(a, b);
}

__m256i test_mm256_cmpgt_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi64
  // CHECK: icmp sgt <4 x i64>
  return _mm256_cmpgt_epi64(a, b);
}

__m256i test_mm256_cvtepi8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi8_epi16
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  return _mm256_cvtepi8_epi16(a);
}

__m256i test_mm256_cvtepi8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi8_epi32
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i32>
  return _mm256_cvtepi8_epi32(a);
}

__m256i test_mm256_cvtepi8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi8_epi64
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sext <4 x i8> %{{.*}} to <4 x i64>
  return _mm256_cvtepi8_epi64(a);
}

__m256i test_mm256_cvtepi16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi16_epi32
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i32>
  return _mm256_cvtepi16_epi32(a);
}

__m256i test_mm256_cvtepi16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi16_epi64
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sext <4 x i16> %{{.*}} to <4 x i64>
  return _mm256_cvtepi16_epi64(a);
}

__m256i test_mm256_cvtepi32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepi32_epi64
  // CHECK: sext <4 x i32> %{{.*}} to <4 x i64>
  return _mm256_cvtepi32_epi64(a);
}

__m256i test_mm256_cvtepu8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu8_epi16
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  return _mm256_cvtepu8_epi16(a);
}

__m256i test_mm256_cvtepu8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu8_epi32
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i32>
  return _mm256_cvtepu8_epi32(a);
}

__m256i test_mm256_cvtepu8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu8_epi64
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: zext <4 x i8> %{{.*}} to <4 x i64>
  return _mm256_cvtepu8_epi64(a);
}

__m256i test_mm256_cvtepu16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu16_epi32
  // CHECK: zext <8 x i16> {{.*}} to <8 x i32>
  return _mm256_cvtepu16_epi32(a);
}

__m256i test_mm256_cvtepu16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu16_epi64
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: zext <4 x i16> %{{.*}} to <4 x i64>
  return _mm256_cvtepu16_epi64(a);
}

__m256i test_mm256_cvtepu32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtepu32_epi64
  // CHECK: zext <4 x i32> %{{.*}} to <4 x i64>
  return _mm256_cvtepu32_epi64(a);
}

__m128i test0_mm256_extracti128_si256_0(__m256i a) {
  // CHECK-LABEL: test0_mm256_extracti128_si256
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> zeroinitializer, <2 x i32> <i32 0, i32 1>
  return _mm256_extracti128_si256(a, 0);
}

__m128i test1_mm256_extracti128_si256_1(__m256i a) {
  // CHECK-LABEL: test1_mm256_extracti128_si256
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> zeroinitializer, <2 x i32> <i32 2, i32 3>
  return _mm256_extracti128_si256(a, 1);
}

// Immediate should be truncated to one bit.
__m128i test2_mm256_extracti128_si256(__m256i a) {
  // CHECK-LABEL: test2_mm256_extracti128_si256
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> zeroinitializer, <2 x i32> <i32 0, i32 1>
  return _mm256_extracti128_si256(a, 2);
}

__m256i test_mm256_hadd_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hadd_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.phadd.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_hadd_epi16(a, b);
}

__m256i test_mm256_hadd_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hadd_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.phadd.d(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_hadd_epi32(a, b);
}

__m256i test_mm256_hadds_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hadds_epi16
  // CHECK:call <16 x i16> @llvm.x86.avx2.phadd.sw(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_hadds_epi16(a, b);
}

__m256i test_mm256_hsub_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hsub_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.phsub.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_hsub_epi16(a, b);
}

__m256i test_mm256_hsub_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hsub_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.phsub.d(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_hsub_epi32(a, b);
}

__m256i test_mm256_hsubs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_hsubs_epi16
  // CHECK:call <16 x i16> @llvm.x86.avx2.phsub.sw(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_hsubs_epi16(a, b);
}

__m128i test_mm_i32gather_epi32(int const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i32gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.d.d(<4 x i32> %{{.*}}, i8* %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_i32gather_epi32(b, c, 2);
}

__m128i test_mm_mask_i32gather_epi32(__m128i a, int const *b, __m128i c, __m128i d) {
  // CHECK-LABEL: test_mm_mask_i32gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.d.d(<4 x i32> %{{.*}}, i8* %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_mask_i32gather_epi32(a, b, c, d, 2);
}

__m256i test_mm256_i32gather_epi32(int const *b, __m256i c) {
  // CHECK-LABEL: test_mm256_i32gather_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.gather.d.d.256(<8 x i32> %{{.*}}, i8* %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, i8 2)
  return _mm256_i32gather_epi32(b, c, 2);
}

__m256i test_mm256_mask_i32gather_epi32(__m256i a, int const *b, __m256i c, __m256i d) {
  // CHECK-LABEL: test_mm256_mask_i32gather_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.gather.d.d.256(<8 x i32> %{{.*}}, i8* %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, i8 2)
  return _mm256_mask_i32gather_epi32(a, b, c, d, 2);
}

__m128i test_mm_i32gather_epi64(long long const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i32gather_epi64
  // CHECK: call <2 x i64> @llvm.x86.avx2.gather.d.q(<2 x i64> zeroinitializer, i8* %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_i32gather_epi64(b, c, 2);
}

__m128i test_mm_mask_i32gather_epi64(__m128i a, long long const *b, __m128i c, __m128i d) {
  // CHECK-LABEL: test_mm_mask_i32gather_epi64
  // CHECK: call <2 x i64> @llvm.x86.avx2.gather.d.q(<2 x i64> %{{.*}}, i8* %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_mask_i32gather_epi64(a, b, c, d, 2);
}

__m256i test_mm256_i32gather_epi64(long long const *b, __m128i c) {
  // CHECK-LABEL: test_mm256_i32gather_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> zeroinitializer, i8* %{{.*}}, <4 x i32> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  return _mm256_i32gather_epi64(b, c, 2);
}

__m256i test_mm256_mask_i32gather_epi64(__m256i a, long long const *b, __m128i c, __m256i d) {
  // CHECK-LABEL: test_mm256_mask_i32gather_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> %{{.*}}, i8* %{{.*}}, <4 x i32> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  return _mm256_mask_i32gather_epi64(a, b, c, d, 2);
}

__m128d test_mm_i32gather_pd(double const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i32gather_pd
  // CHECK:         [[CMP:%.*]] = fcmp oeq <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK: call <2 x double> @llvm.x86.avx2.gather.d.pd(<2 x double> zeroinitializer, i8* %{{.*}}, <4 x i32> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_i32gather_pd(b, c, 2);
}

__m128d test_mm_mask_i32gather_pd(__m128d a, double const *b, __m128i c, __m128d d) {
  // CHECK-LABEL: test_mm_mask_i32gather_pd
  // CHECK: call <2 x double> @llvm.x86.avx2.gather.d.pd(<2 x double> %{{.*}}, i8* %{{.*}}, <4 x i32> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_mask_i32gather_pd(a, b, c, d, 2);
}

__m256d test_mm256_i32gather_pd(double const *b, __m128i c) {
  // CHECK-LABEL: test_mm256_i32gather_pd
  // CHECK:         [[CMP:%.*]] = fcmp oeq <4 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i64> [[SEXT]] to <4 x double>
  // CHECK: call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> zeroinitializer, i8* %{{.*}}, <4 x i32> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  return _mm256_i32gather_pd(b, c, 2);
}

__m256d test_mm256_mask_i32gather_pd(__m256d a, double const *b, __m128i c, __m256d d) {
  // CHECK-LABEL: test_mm256_mask_i32gather_pd
  // CHECK: call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> %{{.*}}, i8* %{{.*}}, <4 x i32> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  return _mm256_mask_i32gather_pd(a, b, c, d, 2);
}

__m128 test_mm_i32gather_ps(float const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i32gather_ps
  // CHECK:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK: call <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> zeroinitializer, i8* %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_i32gather_ps(b, c, 2);
}

__m128 test_mm_mask_i32gather_ps(__m128 a, float const *b, __m128i c, __m128 d) {
  // CHECK-LABEL: test_mm_mask_i32gather_ps
  // CHECK: call <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> %{{.*}}, i8* %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_mask_i32gather_ps(a, b, c, d, 2);
}

__m256 test_mm256_i32gather_ps(float const *b, __m256i c) {
  // CHECK-LABEL: test_mm256_i32gather_ps
  // CHECK:         [[CMP:%.*]] = fcmp oeq <8 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <8 x i1> [[CMP]] to <8 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <8 x i32> [[SEXT]] to <8 x float>
  // CHECK: call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> zeroinitializer, i8* %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}}, i8 2)
  return _mm256_i32gather_ps(b, c, 2);
}

__m256 test_mm256_mask_i32gather_ps(__m256 a, float const *b, __m256i c, __m256 d) {
  // CHECK-LABEL: test_mm256_mask_i32gather_ps
  // CHECK: call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> %{{.*}}, i8* %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}}, i8 2)
  return _mm256_mask_i32gather_ps(a, b, c, d, 2);
}

__m128i test_mm_i64gather_epi32(int const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i64gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.q.d(<4 x i32> %{{.*}}, i8* %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_i64gather_epi32(b, c, 2);
}

__m128i test_mm_mask_i64gather_epi32(__m128i a, int const *b, __m128i c, __m128i d) {
  // CHECK-LABEL: test_mm_mask_i64gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.q.d(<4 x i32> %{{.*}}, i8* %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_mask_i64gather_epi32(a, b, c, d, 2);
}

__m128i test_mm256_i64gather_epi32(int const *b, __m256i c) {
  // CHECK-LABEL: test_mm256_i64gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> %{{.*}}, i8* %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm256_i64gather_epi32(b, c, 2);
}

__m128i test_mm256_mask_i64gather_epi32(__m128i a, int const *b, __m256i c, __m128i d) {
  // CHECK-LABEL: test_mm256_mask_i64gather_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> %{{.*}}, i8* %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm256_mask_i64gather_epi32(a, b, c, d, 2);
}

__m128i test_mm_i64gather_epi64(long long const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i64gather_epi64
  // CHECK: call <2 x i64> @llvm.x86.avx2.gather.q.q(<2 x i64> zeroinitializer, i8* %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_i64gather_epi64(b, c, 2);
}

__m128i test_mm_mask_i64gather_epi64(__m128i a, long long const *b, __m128i c, __m128i d) {
  // CHECK-LABEL: test_mm_mask_i64gather_epi64
  // CHECK: call <2 x i64> @llvm.x86.avx2.gather.q.q(<2 x i64> %{{.*}}, i8* %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_mask_i64gather_epi64(a, b, c, d, 2);
}

__m256i test_mm256_i64gather_epi64(long long const *b, __m256i c) {
  // CHECK-LABEL: test_mm256_i64gather_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> zeroinitializer, i8* %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  return _mm256_i64gather_epi64(b, c, 2);
}

__m256i test_mm256_mask_i64gather_epi64(__m256i a, long long const *b, __m256i c, __m256i d) {
  // CHECK-LABEL: test_mm256_mask_i64gather_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> %{{.*}}, i8* %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, i8 2)
  return _mm256_mask_i64gather_epi64(a, b, c, d, 2);
}

__m128d test_mm_i64gather_pd(double const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i64gather_pd
  // CHECK:         [[CMP:%.*]] = fcmp oeq <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK: call <2 x double> @llvm.x86.avx2.gather.q.pd(<2 x double> zeroinitializer, i8* %{{.*}}, <2 x i64> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_i64gather_pd(b, c, 2);
}

__m128d test_mm_mask_i64gather_pd(__m128d a, double const *b, __m128i c, __m128d d) {
  // CHECK-LABEL: test_mm_mask_i64gather_pd
  // CHECK: call <2 x double> @llvm.x86.avx2.gather.q.pd(<2 x double> %{{.*}}, i8* %{{.*}}, <2 x i64> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_mask_i64gather_pd(a, b, c, d, 2);
}

__m256d test_mm256_i64gather_pd(double const *b, __m256i c) {
  // CHECK-LABEL: test_mm256_i64gather_pd
  // CHECK:         [[CMP:%.*]] = fcmp oeq <4 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i64> [[SEXT]] to <4 x double>
  // CHECK: call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> zeroinitializer, i8* %{{.*}}, <4 x i64> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  return _mm256_i64gather_pd(b, c, 2);
}

__m256d test_mm256_mask_i64gather_pd(__m256d a, double const *b, __m256i c, __m256d d) {
  // CHECK-LABEL: test_mm256_mask_i64gather_pd
  // CHECK: call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> %{{.*}}, i8* %{{.*}}, <4 x i64> %{{.*}}, <4 x double> %{{.*}}, i8 2)
  return _mm256_mask_i64gather_pd(a, b, c, d, 2);
}

__m128 test_mm_i64gather_ps(float const *b, __m128i c) {
  // CHECK-LABEL: test_mm_i64gather_ps
  // CHECK:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK: call <4 x float> @llvm.x86.avx2.gather.q.ps(<4 x float> zeroinitializer, i8* %{{.*}}, <2 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_i64gather_ps(b, c, 2);
}

__m128 test_mm_mask_i64gather_ps(__m128 a, float const *b, __m128i c, __m128 d) {
  // CHECK-LABEL: test_mm_mask_i64gather_ps
  // CHECK: call <4 x float> @llvm.x86.avx2.gather.q.ps(<4 x float> %{{.*}}, i8* %{{.*}}, <2 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_mask_i64gather_ps(a, b, c, d, 2);
}

__m128 test_mm256_i64gather_ps(float const *b, __m256i c) {
  // CHECK-LABEL: test_mm256_i64gather_ps
  // CHECK:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK: call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> zeroinitializer, i8* %{{.*}}, <4 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm256_i64gather_ps(b, c, 2);
}

__m128 test_mm256_mask_i64gather_ps(__m128 a, float const *b, __m256i c, __m128 d) {
  // CHECK-LABEL: test_mm256_mask_i64gather_ps
  // CHECK: call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> %{{.*}}, i8* %{{.*}}, <4 x i64> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm256_mask_i64gather_ps(a, b, c, d, 2);
}

__m256i test0_mm256_inserti128_si256(__m256i a, __m128i b) {
  // CHECK-LABEL: test0_mm256_inserti128_si256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm256_inserti128_si256(a, b, 0);
}

__m256i test1_mm256_inserti128_si256(__m256i a, __m128i b) {
  // CHECK-LABEL: test1_mm256_inserti128_si256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm256_inserti128_si256(a, b, 1);
}

// Immediate should be truncated to one bit.
__m256i test2_mm256_inserti128_si256(__m256i a, __m128i b) {
  // CHECK-LABEL: test2_mm256_inserti128_si256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm256_inserti128_si256(a, b, 2);
}

__m256i test_mm256_madd_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_madd_epi16
  // CHECK: call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_madd_epi16(a, b);
}

__m256i test_mm256_maddubs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_maddubs_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_maddubs_epi16(a, b);
}

__m128i test_mm_maskload_epi32(int const *a, __m128i m) {
  // CHECK-LABEL: test_mm_maskload_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.maskload.d(i8* %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskload_epi32(a, m);
}

__m256i test_mm256_maskload_epi32(int const *a, __m256i m) {
  // CHECK-LABEL: test_mm256_maskload_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.maskload.d.256(i8* %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskload_epi32(a, m);
}

__m128i test_mm_maskload_epi64(long long const *a, __m128i m) {
  // CHECK-LABEL: test_mm_maskload_epi64
  // CHECK: call <2 x i64> @llvm.x86.avx2.maskload.q(i8* %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskload_epi64(a, m);
}

__m256i test_mm256_maskload_epi64(long long const *a, __m256i m) {
  // CHECK-LABEL: test_mm256_maskload_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.maskload.q.256(i8* %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskload_epi64(a, m);
}

void test_mm_maskstore_epi32(int *a, __m128i m, __m128i b) {
  // CHECK-LABEL: test_mm_maskstore_epi32
  // CHECK: call void @llvm.x86.avx2.maskstore.d(i8* %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  _mm_maskstore_epi32(a, m, b);
}

void test_mm256_maskstore_epi32(int *a, __m256i m, __m256i b) {
  // CHECK-LABEL: test_mm256_maskstore_epi32
  // CHECK: call void @llvm.x86.avx2.maskstore.d.256(i8* %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  _mm256_maskstore_epi32(a, m, b);
}

void test_mm_maskstore_epi64(long long *a, __m128i m, __m128i b) {
  // CHECK-LABEL: test_mm_maskstore_epi64
  // CHECK: call void @llvm.x86.avx2.maskstore.q(i8* %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  _mm_maskstore_epi64(a, m, b);
}

void test_mm256_maskstore_epi64(long long *a, __m256i m, __m256i b) {
  // CHECK-LABEL: test_mm256_maskstore_epi64
  // CHECK: call void @llvm.x86.avx2.maskstore.q.256(i8* %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  _mm256_maskstore_epi64(a, m, b);
}

__m256i test_mm256_max_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epi8
  // CHECK:       [[CMP:%.*]] = icmp sgt <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  return _mm256_max_epi8(a, b);
}

__m256i test_mm256_max_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epi16
  // CHECK:       [[CMP:%.*]] = icmp sgt <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  return _mm256_max_epi16(a, b);
}

__m256i test_mm256_max_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epi32
  // CHECK:       [[CMP:%.*]] = icmp sgt <8 x i32> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <8 x i1> [[CMP]], <8 x i32> [[X]], <8 x i32> [[Y]]
  return _mm256_max_epi32(a, b);
}

__m256i test_mm256_max_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epu8
  // CHECK:       [[CMP:%.*]] = icmp ugt <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  return _mm256_max_epu8(a, b);
}

__m256i test_mm256_max_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epu16
  // CHECK:       [[CMP:%.*]] = icmp ugt <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  return _mm256_max_epu16(a, b);
}

__m256i test_mm256_max_epu32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_max_epu32
  // CHECK:       [[CMP:%.*]] = icmp ugt <8 x i32> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <8 x i1> [[CMP]], <8 x i32> [[X]], <8 x i32> [[Y]]
  return _mm256_max_epu32(a, b);
}

__m256i test_mm256_min_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epi8
  // CHECK:       [[CMP:%.*]] = icmp slt <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  return _mm256_min_epi8(a, b);
}

__m256i test_mm256_min_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epi16
  // CHECK:       [[CMP:%.*]] = icmp slt <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  return _mm256_min_epi16(a, b);
}

__m256i test_mm256_min_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epi32
  // CHECK:       [[CMP:%.*]] = icmp slt <8 x i32> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <8 x i1> [[CMP]], <8 x i32> [[X]], <8 x i32> [[Y]]
  return _mm256_min_epi32(a, b);
}

__m256i test_mm256_min_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epu8
  // CHECK:       [[CMP:%.*]] = icmp ult <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  return _mm256_min_epu8(a, b);
}

__m256i test_mm256_min_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epu16
  // CHECK:       [[CMP:%.*]] = icmp ult <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  return _mm256_min_epu16(a, b);
}

__m256i test_mm256_min_epu32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_min_epu32
  // CHECK:       [[CMP:%.*]] = icmp ult <8 x i32> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <8 x i1> [[CMP]], <8 x i32> [[X]], <8 x i32> [[Y]]
  return _mm256_min_epu32(a, b);
}

int test_mm256_movemask_epi8(__m256i a) {
  // CHECK-LABEL: test_mm256_movemask_epi8
  // CHECK: call i32 @llvm.x86.avx2.pmovmskb(<32 x i8> %{{.*}})
  return _mm256_movemask_epi8(a);
}

__m256i test_mm256_mpsadbw_epu8(__m256i x, __m256i y) {
  // CHECK-LABEL: test_mm256_mpsadbw_epu8
  // CHECK: call <16 x i16> @llvm.x86.avx2.mpsadbw(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, i8 3)
  return _mm256_mpsadbw_epu8(x, y, 3);
}

__m256i test_mm256_mul_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mul_epi32
  // CHECK: shl <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  // CHECK: ashr <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  // CHECK: shl <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  // CHECK: ashr <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  // CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mul_epi32(a, b);
}

__m256i test_mm256_mul_epu32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mul_epu32
  // CHECK: and <4 x i64> %{{.*}}, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>
  // CHECK: and <4 x i64> %{{.*}}, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>
  // CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mul_epu32(a, b);
}

__m256i test_mm256_mulhi_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mulhi_epu16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pmulhu.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mulhi_epu16(a, b);
}

__m256i test_mm256_mulhi_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mulhi_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pmulh.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mulhi_epi16(a, b);
}

__m256i test_mm256_mulhrs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mulhrs_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pmul.hr.sw(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mulhrs_epi16(a, b);
}

__m256i test_mm256_mullo_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mullo_epi16
  // CHECK: mul <16 x i16>
  return _mm256_mullo_epi16(a, b);
}

__m256i test_mm256_mullo_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_mullo_epi32
  // CHECK: mul <8 x i32>
  return _mm256_mullo_epi32(a, b);
}

__m256i test_mm256_or_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_or_si256
  // CHECK: or <4 x i64>
  return _mm256_or_si256(a, b);
}

__m256i test_mm256_packs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_packs_epi16
  // CHECK: call <32 x i8> @llvm.x86.avx2.packsswb(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_packs_epi16(a, b);
}

__m256i test_mm256_packs_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_packs_epi32
  // CHECK: call <16 x i16> @llvm.x86.avx2.packssdw(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_packs_epi32(a, b);
}

__m256i test_mm256_packs_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_packs_epu16
  // CHECK:  call <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_packus_epi16(a, b);
}

__m256i test_mm256_packs_epu32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_packs_epu32
  // CHECK: call <16 x i16> @llvm.x86.avx2.packusdw(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_packus_epi32(a, b);
}

__m256i test_mm256_permute2x128_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_permute2x128_si256
  // CHECK: shufflevector <4 x i64> zeroinitializer, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  return _mm256_permute2x128_si256(a, b, 0x38);
}

__m256i test_mm256_permute4x64_epi64(__m256i a) {
  // CHECK-LABEL: test_mm256_permute4x64_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> zeroinitializer, <4 x i32> <i32 3, i32 0, i32 2, i32 0>
  return _mm256_permute4x64_epi64(a, 35);
}

__m256d test_mm256_permute4x64_pd(__m256d a) {
  // CHECK-LABEL: test_mm256_permute4x64_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> zeroinitializer, <4 x i32> <i32 1, i32 2, i32 1, i32 0>
  return _mm256_permute4x64_pd(a, 25);
}

__m256i test_mm256_permutevar8x32_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_permutevar8x32_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_permutevar8x32_epi32(a, b);
}

__m256 test_mm256_permutevar8x32_ps(__m256 a, __m256i b) {
  // CHECK-LABEL: test_mm256_permutevar8x32_ps
  // CHECK: call <8 x float> @llvm.x86.avx2.permps(<8 x float> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_permutevar8x32_ps(a, b);
}

__m256i test_mm256_sad_epu8(__m256i x, __m256i y) {
  // CHECK-LABEL: test_mm256_sad_epu8
  // CHECK: call <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_sad_epu8(x, y);
}

__m256i test_mm256_shuffle_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_shuffle_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_shuffle_epi8(a, b);
}

__m256i test_mm256_shuffle_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_shuffle_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 3, i32 3, i32 0, i32 0, i32 7, i32 7, i32 4, i32 4>
  return _mm256_shuffle_epi32(a, 15);
}

__m256i test_mm256_shufflehi_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_shufflehi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 6, i32 5, i32 8, i32 9, i32 10, i32 11, i32 15, i32 14, i32 14, i32 13>
  return _mm256_shufflehi_epi16(a, 107);
}

__m256i test_mm256_shufflelo_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_shufflelo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 3, i32 0, i32 1, i32 1, i32 4, i32 5, i32 6, i32 7, i32 11, i32 8, i32 9, i32 9, i32 12, i32 13, i32 14, i32 15>
  return _mm256_shufflelo_epi16(a, 83);
}

__m256i test_mm256_sign_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sign_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_sign_epi8(a, b);
}

__m256i test_mm256_sign_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sign_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psign.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_sign_epi16(a, b);
}

__m256i test_mm256_sign_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sign_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psign.d(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_sign_epi32(a, b);
}

__m256i test_mm256_slli_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_slli_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.pslli.w(<16 x i16> %{{.*}}, i32 %{{.*}})
  return _mm256_slli_epi16(a, 3);
}

__m256i test_mm256_slli_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_slli_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.pslli.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_slli_epi32(a, 3);
}

__m256i test_mm256_slli_epi64(__m256i a) {
  // CHECK-LABEL: test_mm256_slli_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.pslli.q(<4 x i64> %{{.*}}, i32 %{{.*}})
  return _mm256_slli_epi64(a, 3);
}

__m256i test_mm256_slli_si256(__m256i a) {
  // CHECK-LABEL: test_mm256_slli_si256
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60>
  return _mm256_slli_si256(a, 3);
}

__m128i test_mm_sllv_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sllv_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.psllv.d(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sllv_epi32(a, b);
}

__m256i test_mm256_sllv_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sllv_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psllv.d.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_sllv_epi32(a, b);
}

__m128i test_mm_sllv_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sllv_epi64
  // CHECK: call <2 x i64> @llvm.x86.avx2.psllv.q(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_sllv_epi64(a, b);
}

__m256i test_mm256_sllv_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sllv_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.psllv.q.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_sllv_epi64(a, b);
}

__m256i test_mm256_sra_epi16(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_sra_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psra.w(<16 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm256_sra_epi16(a, b);
}

__m256i test_mm256_sra_epi32(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_sra_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psra.d(<8 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm256_sra_epi32(a, b);
}

__m256i test_mm256_srai_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_srai_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psrai.w(<16 x i16> %{{.*}}, i32 %{{.*}})
  return _mm256_srai_epi16(a, 3);
}

__m256i test_mm256_srai_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_srai_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrai.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_srai_epi32(a, 3);
}

__m128i test_mm_srav_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_srav_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.psrav.d(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_srav_epi32(a, b);
}

__m256i test_mm256_srav_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_srav_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrav.d.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_srav_epi32(a, b);
}

__m256i test_mm256_srl_epi16(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_srl_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psrl.w(<16 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm256_srl_epi16(a, b);
}

__m256i test_mm256_srl_epi32(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_srl_epi32
  // CHECK:call <8 x i32> @llvm.x86.avx2.psrl.d(<8 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm256_srl_epi32(a, b);
}

__m256i test_mm256_srl_epi64(__m256i a, __m128i b) {
  // CHECK-LABEL: test_mm256_srl_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.psrl.q(<4 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm256_srl_epi64(a, b);
}

__m256i test_mm256_srli_epi16(__m256i a) {
  // CHECK-LABEL: test_mm256_srli_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx2.psrli.w(<16 x i16> %{{.*}}, i32 %{{.*}})
  return _mm256_srli_epi16(a, 3);
}

__m256i test_mm256_srli_epi32(__m256i a) {
  // CHECK-LABEL: test_mm256_srli_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrli.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_srli_epi32(a, 3);
}

__m256i test_mm256_srli_epi64(__m256i a) {
  // CHECK-LABEL: test_mm256_srli_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.psrli.q(<4 x i64> %{{.*}}, i32 %{{.*}})
  return _mm256_srli_epi64(a, 3);
}

__m256i test_mm256_srli_si256(__m256i a) {
  // CHECK-LABEL: test_mm256_srli_si256
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50>
  return _mm256_srli_si256(a, 3);
}

__m128i test_mm_srlv_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_srlv_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx2.psrlv.d(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_srlv_epi32(a, b);
}

__m256i test_mm256_srlv_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_srlv_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx2.psrlv.d.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_srlv_epi32(a, b);
}

__m128i test_mm_srlv_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_srlv_epi64
  // CHECK: call <2 x i64> @llvm.x86.avx2.psrlv.q(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_srlv_epi64(a, b);
}

__m256i test_mm256_srlv_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_srlv_epi64
  // CHECK: call <4 x i64> @llvm.x86.avx2.psrlv.q.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_srlv_epi64(a, b);
}

__m256i test_mm256_stream_load_si256(__m256i const *a) {
  // CHECK-LABEL: test_mm256_stream_load_si256
  // CHECK: load <4 x i64>, <4 x i64>* %{{.*}}, align 32, !nontemporal
  return _mm256_stream_load_si256(a);
}

__m256i test_mm256_sub_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sub_epi8
  // CHECK: sub <32 x i8>
  return _mm256_sub_epi8(a, b);
}

__m256i test_mm256_sub_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sub_epi16
  // CHECK: sub <16 x i16>
  return _mm256_sub_epi16(a, b);
}

__m256i test_mm256_sub_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sub_epi32
  // CHECK: sub <8 x i32>
  return _mm256_sub_epi32(a, b);
}

__m256i test_mm256_sub_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_sub_epi64
  // CHECK: sub <4 x i64>
  return _mm256_sub_epi64(a, b);
}

__m256i test_mm256_subs_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_subs_epi8
  // CHECK-NOT: call <32 x i8> @llvm.x86.avx2.psubs.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: sub <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: icmp sle <32 x i16> %{{.*}}, <i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> <i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127, i16 127>
  // CHECK: icmp slt <32 x i16> %{{.*}}, <i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> <i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128, i16 -128>, <32 x i16> %{{.*}}
  // CHECK: trunc <32 x i16> %{{.*}} to <32 x i8>
  return _mm256_subs_epi8(a, b);
}

__m256i test_mm256_subs_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_subs_epi16
  // CHECK-NOT: call <16 x i16> @llvm.x86.avx2.psubs.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: sub <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp sle <16 x i32> %{{.*}}, <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  // CHECK: icmp slt <16 x i32> %{{.*}}, <i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> <i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768, i32 -32768>, <16 x i32> %{{.*}}
  // CHECK: trunc <16 x i32> %{{.*}} to <16 x i16>
  return _mm256_subs_epi16(a, b);
}

__m256i test_mm256_subs_epu8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_subs_epu8
  // CHECK-NOT: call <32 x i8> @llvm.x86.avx2.psubus.b(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: icmp ugt <32 x i8> {{.*}}, {{.*}}
  // CHECK: select <32 x i1> {{.*}}, <32 x i8> {{.*}}, <32 x i8> {{.*}}
  // CHECK: sub <32 x i8> {{.*}}, {{.*}}
  return _mm256_subs_epu8(a, b);
}

__m256i test_mm256_subs_epu16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_subs_epu16
  // CHECK-NOT: call <16 x i16> @llvm.x86.avx2.psubus.w(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: icmp ugt <16 x i16> {{.*}}, {{.*}}
  // CHECK: select <16 x i1> {{.*}}, <16 x i16> {{.*}}, <16 x i16> {{.*}}
  // CHECK: sub <16 x i16> {{.*}}, {{.*}}
  return _mm256_subs_epu16(a, b);
}

__m256i test_mm256_unpackhi_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpackhi_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  return _mm256_unpackhi_epi8(a, b);
}

__m256i test_mm256_unpackhi_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpackhi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  return _mm256_unpackhi_epi16(a, b);
}

__m256i test_mm256_unpackhi_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpackhi_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  return _mm256_unpackhi_epi32(a, b);
}

__m256i test_mm256_unpackhi_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpackhi_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  return _mm256_unpackhi_epi64(a, b);
}

__m256i test_mm256_unpacklo_epi8(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpacklo_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  return _mm256_unpacklo_epi8(a, b);
}

__m256i test_mm256_unpacklo_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpacklo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  return _mm256_unpacklo_epi16(a, b);
}

__m256i test_mm256_unpacklo_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpacklo_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  return _mm256_unpacklo_epi32(a, b);
}

__m256i test_mm256_unpacklo_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_unpacklo_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  return _mm256_unpacklo_epi64(a, b);
}

__m256i test_mm256_xor_si256(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_xor_si256
  // CHECK: xor <4 x i64>
  return _mm256_xor_si256(a, b);
}
