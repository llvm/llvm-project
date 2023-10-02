// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64


#include <immintrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/sse2-intrinsics-fast-isel.ll

__m128i test_mm_add_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_add_epi8
  // CHECK: add <16 x i8>
  return _mm_add_epi8(A, B);
}

__m128i test_mm_add_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_add_epi16
  // CHECK: add <8 x i16>
  return _mm_add_epi16(A, B);
}

__m128i test_mm_add_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_add_epi32
  // CHECK: add <4 x i32>
  return _mm_add_epi32(A, B);
}

__m128i test_mm_add_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_add_epi64
  // CHECK: add <2 x i64>
  return _mm_add_epi64(A, B);
}

__m128d test_mm_add_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_add_pd
  // CHECK: fadd <2 x double>
  return _mm_add_pd(A, B);
}

__m128d test_mm_add_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_add_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fadd double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_add_sd(A, B);
}

__m128i test_mm_adds_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_adds_epi8
  // CHECK: call <16 x i8> @llvm.sadd.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_adds_epi8(A, B);
}

__m128i test_mm_adds_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_adds_epi16
  // CHECK: call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_adds_epi16(A, B);
}

__m128i test_mm_adds_epu8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_adds_epu8
  // CHECK-NOT: call <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: call <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_adds_epu8(A, B);
}

__m128i test_mm_adds_epu16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_adds_epu16
  // CHECK-NOT: call <8 x i16> @llvm.x86.sse2.paddus.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: call <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_adds_epu16(A, B);
}

__m128d test_mm_and_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_and_pd
  // CHECK: and <2 x i64>
  return _mm_and_pd(A, B);
}

__m128i test_mm_and_si128(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_and_si128
  // CHECK: and <2 x i64>
  return _mm_and_si128(A, B);
}

__m128d test_mm_andnot_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_andnot_pd
  // CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  // CHECK: and <2 x i64>
  return _mm_andnot_pd(A, B);
}

__m128i test_mm_andnot_si128(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_andnot_si128
  // CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  // CHECK: and <2 x i64>
  return _mm_andnot_si128(A, B);
}

__m128i test_mm_avg_epu8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_avg_epu8
  // CHECK: call <16 x i8> @llvm.x86.sse2.pavg.b(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_avg_epu8(A, B);
}

__m128i test_mm_avg_epu16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_avg_epu16
  // CHECK: call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_avg_epu16(A, B);
}

__m128i test_mm_bslli_si128(__m128i A) {
  // CHECK-LABEL: test_mm_bslli_si128
  // CHECK: shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
  return _mm_bslli_si128(A, 5);
}

__m128i test_mm_bsrli_si128(__m128i A) {
  // CHECK-LABEL: test_mm_bsrli_si128
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> zeroinitializer, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  return _mm_bsrli_si128(A, 5);
}

__m128 test_mm_castpd_ps(__m128d A) {
  // CHECK-LABEL: test_mm_castpd_ps
  return _mm_castpd_ps(A);
}

__m128i test_mm_castpd_si128(__m128d A) {
  // CHECK-LABEL: test_mm_castpd_si128
  return _mm_castpd_si128(A);
}

__m128d test_mm_castps_pd(__m128 A) {
  // CHECK-LABEL: test_mm_castps_pd
  return _mm_castps_pd(A);
}

__m128i test_mm_castps_si128(__m128 A) {
  // CHECK-LABEL: test_mm_castps_si128
  return _mm_castps_si128(A);
}

__m128d test_mm_castsi128_pd(__m128i A) {
  // CHECK-LABEL: test_mm_castsi128_pd
  return _mm_castsi128_pd(A);
}

__m128 test_mm_castsi128_ps(__m128i A) {
  // CHECK-LABEL: test_mm_castsi128_ps
  return _mm_castsi128_ps(A);
}

void test_mm_clflush(void* A) {
  // CHECK-LABEL: test_mm_clflush
  // CHECK: call void @llvm.x86.sse2.clflush(ptr %{{.*}})
  _mm_clflush(A);
}

__m128i test_mm_cmpeq_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpeq_epi8
  // CHECK: icmp eq <16 x i8>
  return _mm_cmpeq_epi8(A, B);
}

__m128i test_mm_cmpeq_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpeq_epi16
  // CHECK: icmp eq <8 x i16>
  return _mm_cmpeq_epi16(A, B);
}

__m128i test_mm_cmpeq_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpeq_epi32
  // CHECK: icmp eq <4 x i32>
  return _mm_cmpeq_epi32(A, B);
}

__m128d test_mm_cmpeq_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpeq_pd
  // CHECK:         [[CMP:%.*]] = fcmp oeq <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpeq_pd(A, B);
}

__m128d test_mm_cmpeq_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpeq_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 0)
  return _mm_cmpeq_sd(A, B);
}

__m128d test_mm_cmpge_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpge_pd
  // CHECK:         [[CMP:%.*]] = fcmp ole <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpge_pd(A, B);
}

__m128d test_mm_cmpge_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpge_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 1
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_cmpge_sd(A, B);
}

__m128i test_mm_cmpgt_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpgt_epi8
  // CHECK: icmp sgt <16 x i8>
  return _mm_cmpgt_epi8(A, B);
}

__m128i test_mm_cmpgt_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpgt_epi16
  // CHECK: icmp sgt <8 x i16>
  return _mm_cmpgt_epi16(A, B);
}

__m128i test_mm_cmpgt_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpgt_epi32
  // CHECK: icmp sgt <4 x i32>
  return _mm_cmpgt_epi32(A, B);
}

__m128d test_mm_cmpgt_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpgt_pd
  // CHECK:         [[CMP:%.*]] = fcmp olt <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpgt_pd(A, B);
}

__m128d test_mm_cmpgt_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpgt_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 1
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_cmpgt_sd(A, B);
}

__m128d test_mm_cmple_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmple_pd
  // CHECK:         [[CMP:%.*]] = fcmp ole <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmple_pd(A, B);
}

__m128d test_mm_cmple_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmple_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_cmple_sd(A, B);
}

__m128i test_mm_cmplt_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmplt_epi8
  // CHECK: icmp sgt <16 x i8>
  return _mm_cmplt_epi8(A, B);
}

__m128i test_mm_cmplt_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmplt_epi16
  // CHECK: icmp sgt <8 x i16>
  return _mm_cmplt_epi16(A, B);
}

__m128i test_mm_cmplt_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmplt_epi32
  // CHECK: icmp sgt <4 x i32>
  return _mm_cmplt_epi32(A, B);
}

__m128d test_mm_cmplt_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmplt_pd
  // CHECK:         [[CMP:%.*]] = fcmp olt <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmplt_pd(A, B);
}

__m128d test_mm_cmplt_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmplt_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  return _mm_cmplt_sd(A, B);
}

__m128d test_mm_cmpneq_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpneq_pd
  // CHECK:         [[CMP:%.*]] = fcmp une <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpneq_pd(A, B);
}

__m128d test_mm_cmpneq_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpneq_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 4)
  return _mm_cmpneq_sd(A, B);
}

__m128d test_mm_cmpnge_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpnge_pd
  // CHECK:         [[CMP:%.*]] = fcmp ugt <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpnge_pd(A, B);
}

__m128d test_mm_cmpnge_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpnge_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 1
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_cmpnge_sd(A, B);
}

__m128d test_mm_cmpngt_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpngt_pd
  // CHECK:         [[CMP:%.*]] = fcmp uge <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpngt_pd(A, B);
}

__m128d test_mm_cmpngt_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpngt_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 1
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_cmpngt_sd(A, B);
}

__m128d test_mm_cmpnle_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpnle_pd
  // CHECK:         [[CMP:%.*]] = fcmp ugt <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpnle_pd(A, B);
}

__m128d test_mm_cmpnle_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpnle_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  return _mm_cmpnle_sd(A, B);
}

__m128d test_mm_cmpnlt_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpnlt_pd
  // CHECK:         [[CMP:%.*]] = fcmp uge <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpnlt_pd(A, B);
}

__m128d test_mm_cmpnlt_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpnlt_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  return _mm_cmpnlt_sd(A, B);
}

__m128d test_mm_cmpord_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpord_pd
  // CHECK:         [[CMP:%.*]] = fcmp ord <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpord_pd(A, B);
}

__m128d test_mm_cmpord_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpord_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 7)
  return _mm_cmpord_sd(A, B);
}

__m128d test_mm_cmpunord_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpunord_pd
  // CHECK:         [[CMP:%.*]] = fcmp uno <2 x double>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  return _mm_cmpunord_pd(A, B);
}

__m128d test_mm_cmpunord_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpunord_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 3)
  return _mm_cmpunord_sd(A, B);
}

int test_mm_comieq_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_comieq_sd
  // CHECK: call i32 @llvm.x86.sse2.comieq.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_comieq_sd(A, B);
}

int test_mm_comige_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_comige_sd
  // CHECK: call i32 @llvm.x86.sse2.comige.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_comige_sd(A, B);
}

int test_mm_comigt_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_comigt_sd
  // CHECK: call i32 @llvm.x86.sse2.comigt.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_comigt_sd(A, B);
}

int test_mm_comile_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_comile_sd
  // CHECK: call i32 @llvm.x86.sse2.comile.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_comile_sd(A, B);
}

int test_mm_comilt_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_comilt_sd
  // CHECK: call i32 @llvm.x86.sse2.comilt.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_comilt_sd(A, B);
}

int test_mm_comineq_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_comineq_sd
  // CHECK: call i32 @llvm.x86.sse2.comineq.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_comineq_sd(A, B);
}

__m128d test_mm_cvtepi32_pd(__m128i A) {
  // CHECK-LABEL: test_mm_cvtepi32_pd
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: sitofp <2 x i32> %{{.*}} to <2 x double>
  return _mm_cvtepi32_pd(A);
}

__m128 test_mm_cvtepi32_ps(__m128i A) {
  // CHECK-LABEL: test_mm_cvtepi32_ps
  // CHECK: sitofp <4 x i32> %{{.*}} to <4 x float>
  return _mm_cvtepi32_ps(A);
}

__m128i test_mm_cvtpd_epi32(__m128d A) {
  // CHECK-LABEL: test_mm_cvtpd_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.cvtpd2dq(<2 x double> %{{.*}})
  return _mm_cvtpd_epi32(A);
}

__m128 test_mm_cvtpd_ps(__m128d A) {
  // CHECK-LABEL: test_mm_cvtpd_ps
  // CHECK: call <4 x float> @llvm.x86.sse2.cvtpd2ps(<2 x double> %{{.*}})
  return _mm_cvtpd_ps(A);
}

__m128i test_mm_cvtps_epi32(__m128 A) {
  // CHECK-LABEL: test_mm_cvtps_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float> %{{.*}})
  return _mm_cvtps_epi32(A);
}

__m128d test_mm_cvtps_pd(__m128 A) {
  // CHECK-LABEL: test_mm_cvtps_pd
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: fpext <2 x float> %{{.*}} to <2 x double>
  return _mm_cvtps_pd(A);
}

double test_mm_cvtsd_f64(__m128d A) {
  // CHECK-LABEL: test_mm_cvtsd_f64
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  return _mm_cvtsd_f64(A);
}

int test_mm_cvtsd_si32(__m128d A) {
  // CHECK-LABEL: test_mm_cvtsd_si32
  // CHECK: call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %{{.*}})
  return _mm_cvtsd_si32(A);
}

#ifdef __x86_64__
long long test_mm_cvtsd_si64(__m128d A) {
  // X64-LABEL: test_mm_cvtsd_si64
  // X64: call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %{{.*}})
  return _mm_cvtsd_si64(A);
}
#endif

__m128 test_mm_cvtsd_ss(__m128 A, __m128d B) {
  // CHECK-LABEL: test_mm_cvtsd_ss
  // CHECK: call <4 x float> @llvm.x86.sse2.cvtsd2ss(<4 x float> %{{.*}}, <2 x double> %{{.*}})
  return _mm_cvtsd_ss(A, B);
}

int test_mm_cvtsi128_si32(__m128i A) {
  // CHECK-LABEL: test_mm_cvtsi128_si32
  // CHECK: extractelement <4 x i32> %{{.*}}, i32 0
  return _mm_cvtsi128_si32(A);
}

long long test_mm_cvtsi128_si64(__m128i A) {
  // CHECK-LABEL: test_mm_cvtsi128_si64
  // CHECK: extractelement <2 x i64> %{{.*}}, i32 0
  return _mm_cvtsi128_si64(A);
}

__m128d test_mm_cvtsi32_sd(__m128d A, int B) {
  // CHECK-LABEL: test_mm_cvtsi32_sd
  // CHECK: sitofp i32 %{{.*}} to double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_cvtsi32_sd(A, B);
}

__m128i test_mm_cvtsi32_si128(int A) {
  // CHECK-LABEL: test_mm_cvtsi32_si128
  // CHECK: insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 0, i32 1
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 0, i32 2
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 0, i32 3
  return _mm_cvtsi32_si128(A);
}

#ifdef __x86_64__
__m128d test_mm_cvtsi64_sd(__m128d A, long long B) {
  // X64-LABEL: test_mm_cvtsi64_sd
  // X64: sitofp i64 %{{.*}} to double
  // X64: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_cvtsi64_sd(A, B);
}
#endif

__m128i test_mm_cvtsi64_si128(long long A) {
  // CHECK-LABEL: test_mm_cvtsi64_si128
  // CHECK: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 0, i32 1
  return _mm_cvtsi64_si128(A);
}

__m128d test_mm_cvtss_sd(__m128d A, __m128 B) {
  // CHECK-LABEL: test_mm_cvtss_sd
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fpext float %{{.*}} to double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_cvtss_sd(A, B);
}

__m128i test_mm_cvttpd_epi32(__m128d A) {
  // CHECK-LABEL: test_mm_cvttpd_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double> %{{.*}})
  return _mm_cvttpd_epi32(A);
}

__m128i test_mm_cvttps_epi32(__m128 A) {
  // CHECK-LABEL: test_mm_cvttps_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float> %{{.*}})
  return _mm_cvttps_epi32(A);
}

int test_mm_cvttsd_si32(__m128d A) {
  // CHECK-LABEL: test_mm_cvttsd_si32
  // CHECK: call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> %{{.*}})
  return _mm_cvttsd_si32(A);
}

#ifdef __x86_64__
long long test_mm_cvttsd_si64(__m128d A) {
  // X64-LABEL: test_mm_cvttsd_si64
  // X64: call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %{{.*}})
  return _mm_cvttsd_si64(A);
}
#endif

__m128d test_mm_div_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_div_pd
  // CHECK: fdiv <2 x double>
  return _mm_div_pd(A, B);
}

__m128d test_mm_div_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_div_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fdiv double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_div_sd(A, B);
}

// Lowering to pextrw requires optimization.
int test_mm_extract_epi16(__m128i A) {
  // CHECK-LABEL: test_mm_extract_epi16
  // CHECK: extractelement <8 x i16> %{{.*}}, {{i32|i64}} 1
  // CHECK: zext i16 %{{.*}} to i32
  return _mm_extract_epi16(A, 1);
}

__m128i test_mm_insert_epi16(__m128i A, int B) {
  // CHECK-LABEL: test_mm_insert_epi16
  // CHECK: insertelement <8 x i16> %{{.*}}, {{i32|i64}} 0
  return _mm_insert_epi16(A, B, 0);
}

void test_mm_lfence(void) {
  // CHECK-LABEL: test_mm_lfence
  // CHECK: call void @llvm.x86.sse2.lfence()
  _mm_lfence();
}

__m128d test_mm_load_pd(double const* A) {
  // CHECK-LABEL: test_mm_load_pd
  // CHECK: load <2 x double>, ptr %{{.*}}, align 16
  return _mm_load_pd(A);
}

__m128d test_mm_load_pd1(double const* A) {
  // CHECK-LABEL: test_mm_load_pd1
  // CHECK: load double, ptr %{{.*}}, align 8
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_load_pd1(A);
}

__m128d test_mm_load_sd(double const* A) {
  // CHECK-LABEL: test_mm_load_sd
  // CHECK: load double, ptr %{{.*}}, align 1{{$}}
  return _mm_load_sd(A);
}

__m128i test_mm_load_si128(__m128i const* A) {
  // CHECK-LABEL: test_mm_load_si128
  // CHECK: load <2 x i64>, ptr %{{.*}}, align 16
  return _mm_load_si128(A);
}

__m128d test_mm_load1_pd(double const* A) {
  // CHECK-LABEL: test_mm_load1_pd
  // CHECK: load double, ptr %{{.*}}, align 8
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_load1_pd(A);
}

__m128d test_mm_loadh_pd(__m128d x, void* y) {
  // CHECK-LABEL: test_mm_loadh_pd
  // CHECK: load double, ptr %{{.*}}, align 1{{$}}
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_loadh_pd(x, y);
}

__m128i test_mm_loadl_epi64(__m128i* y) {
  // CHECK: test_mm_loadl_epi64
  // CHECK: load i64, ptr {{.*}}, align 1{{$}}
  // CHECK: insertelement <2 x i64> undef, i64 {{.*}}, i32 0
  // CHECK: insertelement <2 x i64> {{.*}}, i64 0, i32 1
  return _mm_loadl_epi64(y);
}

__m128d test_mm_loadl_pd(__m128d x, void* y) {
  // CHECK-LABEL: test_mm_loadl_pd
  // CHECK: load double, ptr %{{.*}}, align 1{{$}}
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 1
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_loadl_pd(x, y);
}

__m128d test_mm_loadr_pd(double const* A) {
  // CHECK-LABEL: test_mm_loadr_pd
  // CHECK: load <2 x double>, ptr %{{.*}}, align 16
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 0>
  return _mm_loadr_pd(A);
}

__m128d test_mm_loadu_pd(double const* A) {
  // CHECK-LABEL: test_mm_loadu_pd
  // CHECK: load <2 x double>, ptr %{{.*}}, align 1{{$}}
  return _mm_loadu_pd(A);
}

__m128i test_mm_loadu_si128(__m128i const* A) {
  // CHECK-LABEL: test_mm_loadu_si128
  // CHECK: load <2 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm_loadu_si128(A);
}

__m128i test_mm_loadu_si64(void const* A) {
  // CHECK-LABEL: test_mm_loadu_si64
  // CHECK: load i64, ptr %{{.*}}, align 1{{$}}
  // CHECK: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 0, i32 1
  return _mm_loadu_si64(A);
}

__m128i test_mm_loadu_si32(void const* A) {
  // CHECK-LABEL: test_mm_loadu_si32
  // CHECK: load i32, ptr %{{.*}}, align 1{{$}}
  // CHECK: insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 0, i32 1
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 0, i32 2
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 0, i32 3
  return _mm_loadu_si32(A);
}

__m128i test_mm_loadu_si16(void const* A) {
  // CHECK-LABEL: test_mm_loadu_si16
  // CHECK: load i16, ptr %{{.*}}, align 1{{$}}
  // CHECK: insertelement <8 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 3
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 4
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 5
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 6
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 7
  return _mm_loadu_si16(A);
}

__m128i test_mm_madd_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_madd_epi16
  // CHECK: call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_madd_epi16(A, B);
}

void test_mm_maskmoveu_si128(__m128i A, __m128i B, char* C) {
  // CHECK-LABEL: test_mm_maskmoveu_si128
  // CHECK: call void @llvm.x86.sse2.maskmov.dqu(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, ptr %{{.*}})
  _mm_maskmoveu_si128(A, B, C);
}

__m128i test_mm_max_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_max_epi16
  // CHECK: call <8 x i16> @llvm.smax.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_max_epi16(A, B);
}

__m128i test_mm_max_epu8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_max_epu8
  // CHECK: call <16 x i8> @llvm.umax.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_max_epu8(A, B);
}

__m128d test_mm_max_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_max_pd
  // CHECK: call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_max_pd(A, B);
}

__m128d test_mm_max_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_max_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.max.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_max_sd(A, B);
}

void test_mm_mfence(void) {
  // CHECK-LABEL: test_mm_mfence
  // CHECK: call void @llvm.x86.sse2.mfence()
  _mm_mfence();
}

__m128i test_mm_min_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_min_epi16
  // CHECK: call <8 x i16> @llvm.smin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_min_epi16(A, B);
}

__m128i test_mm_min_epu8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_min_epu8
  // CHECK: call <16 x i8> @llvm.umin.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_min_epu8(A, B);
}

__m128d test_mm_min_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_min_pd
  // CHECK: call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_min_pd(A, B);
}

__m128d test_mm_min_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_min_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.min.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_min_sd(A, B);
}

__m64 test_mm_movepi64_pi64(__m128i A)
{
  // CHECK-LABEL: test_mm_movepi64_pi64
  // CHECK: [[EXT:%.*]] = extractelement <2 x i64> %1, i32 0
  return _mm_movepi64_pi64(A);
}

__m128i test_mm_movpi64_epi64(__m64 A)
{
  // CHECK-LABEL: test_mm_movpi64_epi64
  // CHECK: [[CAST:%.*]] = bitcast <1 x i64> %{{.*}} to i64
  // CHECK: [[INS:%.*]] = insertelement <2 x i64> undef, i64 [[CAST]], i32 0
  // CHECK: insertelement <2 x i64> [[INS]], i64 0, i32 1
  return _mm_movpi64_epi64(A);
}

__m128i test_mm_move_epi64(__m128i A) {
  // CHECK-LABEL: test_mm_move_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  return _mm_move_epi64(A);
}

__m128d test_mm_move_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_move_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_move_sd(A, B);
}

int test_mm_movemask_epi8(__m128i A) {
  // CHECK-LABEL: test_mm_movemask_epi8
  // CHECK: call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %{{.*}})
  return _mm_movemask_epi8(A);
}

int test_mm_movemask_pd(__m128d A) {
  // CHECK-LABEL: test_mm_movemask_pd
  // CHECK: call i32 @llvm.x86.sse2.movmsk.pd(<2 x double> %{{.*}})
  return _mm_movemask_pd(A);
}

__m128i test_mm_mul_epu32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_mul_epu32
  // CHECK: and <2 x i64> %{{.*}}, <i64 4294967295, i64 4294967295>
  // CHECK: and <2 x i64> %{{.*}}, <i64 4294967295, i64 4294967295>
  // CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mul_epu32(A, B);
}

__m128d test_mm_mul_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_mul_pd
  // CHECK: fmul <2 x double> %{{.*}}, %{{.*}}
  return _mm_mul_pd(A, B);
}

__m128d test_mm_mul_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_mul_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fmul double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_mul_sd(A, B);
}

__m128i test_mm_mulhi_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_mulhi_epi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_mulhi_epi16(A, B);
}

__m128i test_mm_mulhi_epu16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_mulhi_epu16
  // CHECK: call <8 x i16> @llvm.x86.sse2.pmulhu.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_mulhi_epu16(A, B);
}

__m128i test_mm_mullo_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_mullo_epi16
  // CHECK: mul <8 x i16> %{{.*}}, %{{.*}}
  return _mm_mullo_epi16(A, B);
}

__m128d test_mm_or_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_or_pd
  // CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  return _mm_or_pd(A, B);
}

__m128i test_mm_or_si128(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_or_si128
  // CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  return _mm_or_si128(A, B);
}

__m128i test_mm_packs_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_packs_epi16
  // CHECK: call <16 x i8> @llvm.x86.sse2.packsswb.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_packs_epi16(A, B);
}

__m128i test_mm_packs_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_packs_epi32
  // CHECK: call <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_packs_epi32(A, B);
}

__m128i test_mm_packus_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_packus_epi16
  // CHECK: call <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_packus_epi16(A, B);
}

void test_mm_pause(void) {
  // CHECK-LABEL: test_mm_pause
  // CHECK: call void @llvm.x86.sse2.pause()
  return _mm_pause();
}

__m128i test_mm_sad_epu8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sad_epu8
  // CHECK: call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_sad_epu8(A, B);
}

__m128i test_mm_set_epi8(char A, char B, char C, char D,
                         char E, char F, char G, char H,
                         char I, char J, char K, char L,
                         char M, char N, char O, char P) {
  // CHECK-LABEL: test_mm_set_epi8
  // CHECK: insertelement <16 x i8> undef, i8 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  return _mm_set_epi8(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
}

__m128i test_mm_set_epi16(short A, short B, short C, short D,
                          short E, short F, short G, short H) {
  // CHECK-LABEL: test_mm_set_epi16
  // CHECK: insertelement <8 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  return _mm_set_epi16(A, B, C, D, E, F, G, H);
}

__m128i test_mm_set_epi32(int A, int B, int C, int D) {
  // CHECK-LABEL: test_mm_set_epi32
  // CHECK: insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  return _mm_set_epi32(A, B, C, D);
}

__m128i test_mm_set_epi64(__m64 A, __m64 B) {
  // CHECK-LABEL: test_mm_set_epi64
  // CHECK: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  return _mm_set_epi64(A, B);
}

__m128i test_mm_set_epi64x(long long A, long long B) {
  // CHECK-LABEL: test_mm_set_epi64x
  // CHECK: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  return _mm_set_epi64x(A, B);
}

__m128d test_mm_set_pd(double A, double B) {
  // CHECK-LABEL: test_mm_set_pd
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_set_pd(A, B);
}

__m128d test_mm_set_pd1(double A) {
  // CHECK-LABEL: test_mm_set_pd1
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_set_pd1(A);
}

__m128d test_mm_set_sd(double A) {
  // CHECK-LABEL: test_mm_set_sd
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double 0.000000e+00, i32 1
  return _mm_set_sd(A);
}

__m128i test_mm_set1_epi8(char A) {
  // CHECK-LABEL: test_mm_set1_epi8
  // CHECK: insertelement <16 x i8> undef, i8 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  return _mm_set1_epi8(A);
}

__m128i test_mm_set1_epi16(short A) {
  // CHECK-LABEL: test_mm_set1_epi16
  // CHECK: insertelement <8 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  return _mm_set1_epi16(A);
}

__m128i test_mm_set1_epi32(int A) {
  // CHECK-LABEL: test_mm_set1_epi32
  // CHECK: insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  return _mm_set1_epi32(A);
}

__m128i test_mm_set1_epi64(__m64 A) {
  // CHECK-LABEL: test_mm_set1_epi64
  // CHECK: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  return _mm_set1_epi64(A);
}

__m128i test_mm_set1_epi64x(long long A) {
  // CHECK-LABEL: test_mm_set1_epi64x
  // CHECK: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  return _mm_set1_epi64x(A);
}

__m128d test_mm_set1_pd(double A) {
  // CHECK-LABEL: test_mm_set1_pd
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_set1_pd(A);
}

__m128i test_mm_setr_epi8(char A, char B, char C, char D,
                          char E, char F, char G, char H,
                          char I, char J, char K, char L,
                          char M, char N, char O, char P) {
  // CHECK-LABEL: test_mm_setr_epi8
  // CHECK: insertelement <16 x i8> undef, i8 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  return _mm_setr_epi8(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
}

__m128i test_mm_setr_epi16(short A, short B, short C, short D,
                           short E, short F, short G, short H) {
  // CHECK-LABEL: test_mm_setr_epi16
  // CHECK: insertelement <8 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  return _mm_setr_epi16(A, B, C, D, E, F, G, H);
}

__m128i test_mm_setr_epi32(int A, int B, int C, int D) {
  // CHECK-LABEL: test_mm_setr_epi32
  // CHECK: insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  return _mm_setr_epi32(A, B, C, D);
}

__m128i test_mm_setr_epi64(__m64 A, __m64 B) {
  // CHECK-LABEL: test_mm_setr_epi64
  // CHECK: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  return _mm_setr_epi64(A, B);
}

__m128d test_mm_setr_pd(double A, double B) {
  // CHECK-LABEL: test_mm_setr_pd
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_setr_pd(A, B);
}

__m128d test_mm_setzero_pd(void) {
  // CHECK-LABEL: test_mm_setzero_pd
  // CHECK: store <2 x double> zeroinitializer
  return _mm_setzero_pd();
}

__m128i test_mm_setzero_si128(void) {
  // CHECK-LABEL: test_mm_setzero_si128
  // CHECK: store <2 x i64> zeroinitializer
  return _mm_setzero_si128();
}

__m128i test_mm_shuffle_epi32(__m128i A) {
  // CHECK-LABEL: test_mm_shuffle_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> zeroinitializer
  return _mm_shuffle_epi32(A, 0);
}

__m128d test_mm_shuffle_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_shuffle_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 2>
  return _mm_shuffle_pd(A, B, 1);
}

__m128i test_mm_shufflehi_epi16(__m128i A) {
  // CHECK-LABEL: test_mm_shufflehi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  return _mm_shufflehi_epi16(A, 0);
}

__m128i test_mm_shufflelo_epi16(__m128i A) {
  // CHECK-LABEL: test_mm_shufflelo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>
  return _mm_shufflelo_epi16(A, 0);
}

__m128i test_mm_sll_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sll_epi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psll.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_sll_epi16(A, B);
}

__m128i test_mm_sll_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sll_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sll_epi32(A, B);
}

__m128i test_mm_sll_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sll_epi64
  // CHECK: call <2 x i64> @llvm.x86.sse2.psll.q(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_sll_epi64(A, B);
}

__m128i test_mm_slli_epi16(__m128i A) {
  // CHECK-LABEL: test_mm_slli_epi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16> %{{.*}}, i32 %{{.*}})
  return _mm_slli_epi16(A, 1);
}

__m128i test_mm_slli_epi16_1(__m128i A) {
  // CHECK-LABEL: test_mm_slli_epi16_1
  // CHECK: call <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16> %{{.*}}, i32 %{{.*}})
  return _mm_slli_epi16(A, -1);
}

__m128i test_mm_slli_epi16_2(__m128i A, int B) {
  // CHECK-LABEL: test_mm_slli_epi16_2
  // CHECK: call <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16> %{{.*}}, i32 %{{.*}})
  return _mm_slli_epi16(A, B);
}

__m128i test_mm_slli_epi32(__m128i A) {
  // CHECK-LABEL: test_mm_slli_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_slli_epi32(A, 1);
}

__m128i test_mm_slli_epi32_1(__m128i A) {
  // CHECK-LABEL: test_mm_slli_epi32_1
  // CHECK: call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_slli_epi32(A, -1);
}

__m128i test_mm_slli_epi32_2(__m128i A, int B) {
  // CHECK-LABEL: test_mm_slli_epi32_2
  // CHECK: call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_slli_epi32(A, B);
}

__m128i test_mm_slli_epi64(__m128i A) {
  // CHECK-LABEL: test_mm_slli_epi64
  // CHECK: call <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64> %{{.*}}, i32 %{{.*}})
  return _mm_slli_epi64(A, 1);
}

__m128i test_mm_slli_epi64_1(__m128i A) {
  // CHECK-LABEL: test_mm_slli_epi64_1
  // CHECK: call <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64> %{{.*}}, i32 %{{.*}})
  return _mm_slli_epi64(A, -1);
}

__m128i test_mm_slli_epi64_2(__m128i A, int B) {
  // CHECK-LABEL: test_mm_slli_epi64_2
  // CHECK: call <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64> %{{.*}}, i32 %{{.*}})
  return _mm_slli_epi64(A, B);
}

__m128i test_mm_slli_si128(__m128i A) {
  // CHECK-LABEL: test_mm_slli_si128
  // CHECK: shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
  return _mm_slli_si128(A, 5);
}

__m128i test_mm_slli_si128_2(__m128i A) {
  // CHECK-LABEL: test_mm_slli_si128_2
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_slli_si128(A, 17);
}

__m128d test_mm_sqrt_pd(__m128d A) {
  // CHECK-LABEL: test_mm_sqrt_pd
  // CHECK: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %{{.*}})
  return _mm_sqrt_pd(A);
}

__m128d test_mm_sqrt_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_sqrt_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: call double @llvm.sqrt.f64(double {{.*}})
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_sqrt_sd(A, B);
}

__m128i test_mm_sra_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sra_epi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_sra_epi16(A, B);
}

__m128i test_mm_sra_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sra_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sra_epi32(A, B);
}

__m128i test_mm_srai_epi16(__m128i A) {
  // CHECK-LABEL: test_mm_srai_epi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> %{{.*}}, i32 %{{.*}})
  return _mm_srai_epi16(A, 1);
}

__m128i test_mm_srai_epi16_1(__m128i A) {
  // CHECK-LABEL: test_mm_srai_epi16_1
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> %{{.*}}, i32 %{{.*}})
  return _mm_srai_epi16(A, -1);
}

__m128i test_mm_srai_epi16_2(__m128i A, int B) {
  // CHECK-LABEL: test_mm_srai_epi16_2
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> %{{.*}}, i32 %{{.*}})
  return _mm_srai_epi16(A, B);
}

__m128i test_mm_srai_epi32(__m128i A) {
  // CHECK-LABEL: test_mm_srai_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_srai_epi32(A, 1);
}

__m128i test_mm_srai_epi32_1(__m128i A) {
  // CHECK-LABEL: test_mm_srai_epi32_1
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_srai_epi32(A, -1);
}

__m128i test_mm_srai_epi32_2(__m128i A, int B) {
  // CHECK-LABEL: test_mm_srai_epi32_2
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_srai_epi32(A, B);
}

__m128i test_mm_srl_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_srl_epi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_srl_epi16(A, B);
}

__m128i test_mm_srl_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_srl_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrl.d(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_srl_epi32(A, B);
}

__m128i test_mm_srl_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_srl_epi64
  // CHECK: call <2 x i64> @llvm.x86.sse2.psrl.q(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_srl_epi64(A, B);
}

__m128i test_mm_srli_epi16(__m128i A) {
  // CHECK-LABEL: test_mm_srli_epi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> %{{.*}}, i32 %{{.*}})
  return _mm_srli_epi16(A, 1);
}

__m128i test_mm_srli_epi16_1(__m128i A) {
  // CHECK-LABEL: test_mm_srli_epi16_1
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> %{{.*}}, i32 %{{.*}})
  return _mm_srli_epi16(A, -1);
}

__m128i test_mm_srli_epi16_2(__m128i A, int B) {
  // CHECK-LABEL: test_mm_srli_epi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> %{{.*}}, i32 %{{.*}})
  return _mm_srli_epi16(A, B);
}

__m128i test_mm_srli_epi32(__m128i A) {
  // CHECK-LABEL: test_mm_srli_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_srli_epi32(A, 1);
}

__m128i test_mm_srli_epi32_1(__m128i A) {
  // CHECK-LABEL: test_mm_srli_epi32_1
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_srli_epi32(A, -1);
}

__m128i test_mm_srli_epi32_2(__m128i A, int B) {
  // CHECK-LABEL: test_mm_srli_epi32_2
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_srli_epi32(A, B);
}

__m128i test_mm_srli_epi64(__m128i A) {
  // CHECK-LABEL: test_mm_srli_epi64
  // CHECK: call <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64> %{{.*}}, i32 %{{.*}})
  return _mm_srli_epi64(A, 1);
}

__m128i test_mm_srli_epi64_1(__m128i A) {
  // CHECK-LABEL: test_mm_srli_epi64_1
  // CHECK: call <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64> %{{.*}}, i32 %{{.*}})
  return _mm_srli_epi64(A, -1);
}

__m128i test_mm_srli_epi64_2(__m128i A, int B) {
  // CHECK-LABEL: test_mm_srli_epi64_2
  // CHECK: call <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64> %{{.*}}, i32 %{{.*}})
  return _mm_srli_epi64(A, B);
}

__m128i test_mm_srli_si128(__m128i A) {
  // CHECK-LABEL: test_mm_srli_si128
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> zeroinitializer, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  return _mm_srli_si128(A, 5);
}

__m128i test_mm_srli_si128_2(__m128i A) {
  // CHECK-LABEL: test_mm_srli_si128_2
  // ret <2 x i64> zeroinitializer
  return _mm_srli_si128(A, 17);
}

void test_mm_store_pd(double* A, __m128d B) {
  // CHECK-LABEL: test_mm_store_pd
  // CHECK: store <2 x double> %{{.*}}, ptr %{{.*}}, align 16
  _mm_store_pd(A, B);
}

void test_mm_store_pd1(double* x, __m128d y) {
  // CHECK-LABEL: test_mm_store_pd1
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: store <2 x double> %{{.*}}, ptr {{.*}}, align 16
  _mm_store_pd1(x, y);
}

void test_mm_store_sd(double* A, __m128d B) {
  // CHECK-LABEL: test_mm_store_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: store double %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm_store_sd(A, B);
}

void test_mm_store_si128(__m128i* A, __m128i B) {
  // CHECK-LABEL: test_mm_store_si128
  // CHECK: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 16
  _mm_store_si128(A, B);
}

void test_mm_store1_pd(double* x, __m128d y) {
  // CHECK-LABEL: test_mm_store1_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: store <2 x double> %{{.*}}, ptr %{{.*}}, align 16
  _mm_store1_pd(x, y);
}

void test_mm_storeh_pd(double* A, __m128d B) {
  // CHECK-LABEL: test_mm_storeh_pd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 1
  // CHECK: store double %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm_storeh_pd(A, B);
}

void test_mm_storel_epi64(__m128i x, void* y) {
  // CHECK-LABEL: test_mm_storel_epi64
  // CHECK: extractelement <2 x i64> %{{.*}}, i32 0
  // CHECK: store {{.*}} ptr {{.*}}, align 1{{$}}
  _mm_storel_epi64(y, x);
}

void test_mm_storel_pd(double* A, __m128d B) {
  // CHECK-LABEL: test_mm_storel_pd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: store double %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm_storel_pd(A, B);
}

void test_mm_storer_pd(__m128d A, double* B) {
  // CHECK-LABEL: test_mm_storer_pd
  // CHECK: shufflevector <2 x double> {{.*}}, <2 x double> {{.*}}, <2 x i32> <i32 1, i32 0>
  // CHECK: store {{.*}} ptr {{.*}}, align 16{{$}}
  _mm_storer_pd(B, A);
}

void test_mm_storeu_pd(double* A, __m128d B) {
  // CHECK-LABEL: test_mm_storeu_pd
  // CHECK: store {{.*}} ptr {{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm_storeu_pd(A, B);
}

void test_mm_storeu_si128(__m128i* A, __m128i B) {
  // CHECK-LABEL: test_mm_storeu_si128
  // CHECK: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm_storeu_si128(A, B);
}

void test_mm_storeu_si64(void* A, __m128i B) {
  // CHECK-LABEL: test_mm_storeu_si64
  // CHECK: [[EXT:%.*]] = extractelement <2 x i64> %{{.*}}, i32 0
  // CHECK: store i64 [[EXT]], ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm_storeu_si64(A, B);
}

void test_mm_storeu_si32(void* A, __m128i B) {
  // CHECK-LABEL: test_mm_storeu_si32
  // CHECK: [[EXT:%.*]] = extractelement <4 x i32> %{{.*}}, i32 0
  // CHECK: store i32 [[EXT]], ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm_storeu_si32(A, B);
}

void test_mm_storeu_si16(void* A, __m128i B) {
  // CHECK-LABEL: test_mm_storeu_si16
  // CHECK: [[EXT:%.*]] = extractelement <8 x i16> %{{.*}}, i32 0
  // CHECK: store i16 [[EXT]], ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm_storeu_si16(A, B);
}

void test_mm_stream_pd(double *A, __m128d B) {
  // CHECK-LABEL: test_mm_stream_pd
  // CHECK: store <2 x double> %{{.*}}, ptr %{{.*}}, align 16, !nontemporal
  _mm_stream_pd(A, B);
}

void test_mm_stream_pd_void(void *A, __m128d B) {
  // CHECK-LABEL: test_mm_stream_pd_void
  // CHECK: store <2 x double> %{{.*}}, ptr %{{.*}}, align 16, !nontemporal
  _mm_stream_pd(A, B);
}

void test_mm_stream_si32(int *A, int B) {
  // CHECK-LABEL: test_mm_stream_si32
  // CHECK: store i32 %{{.*}}, ptr %{{.*}}, align 1, !nontemporal
  _mm_stream_si32(A, B);
}

void test_mm_stream_si32_void(void *A, int B) {
  // CHECK-LABEL: test_mm_stream_si32_void
  // CHECK: store i32 %{{.*}}, ptr %{{.*}}, align 1, !nontemporal
  _mm_stream_si32(A, B);
}

#ifdef __x86_64__
void test_mm_stream_si64(long long *A, long long B) {
  // X64-LABEL: test_mm_stream_si64
  // X64: store i64 %{{.*}}, ptr %{{.*}}, align 1, !nontemporal
  _mm_stream_si64(A, B);
}

void test_mm_stream_si64_void(void *A, long long B) {
  // X64-LABEL: test_mm_stream_si64_void
  // X64: store i64 %{{.*}}, ptr %{{.*}}, align 1, !nontemporal
  _mm_stream_si64(A, B);
}
#endif

void test_mm_stream_si128(__m128i *A, __m128i B) {
  // CHECK-LABEL: test_mm_stream_si128
  // CHECK: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 16, !nontemporal
  _mm_stream_si128(A, B);
}

void test_mm_stream_si128_void(void *A, __m128i B) {
  // CHECK-LABEL: test_mm_stream_si128_void
  // CHECK: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 16, !nontemporal
  _mm_stream_si128(A, B);
}

__m128i test_mm_sub_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sub_epi8
  // CHECK: sub <16 x i8>
  return _mm_sub_epi8(A, B);
}

__m128i test_mm_sub_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sub_epi16
  // CHECK: sub <8 x i16>
  return _mm_sub_epi16(A, B);
}

__m128i test_mm_sub_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sub_epi32
  // CHECK: sub <4 x i32>
  return _mm_sub_epi32(A, B);
}

__m128i test_mm_sub_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_sub_epi64
  // CHECK: sub <2 x i64>
  return _mm_sub_epi64(A, B);
}

__m128d test_mm_sub_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_sub_pd
  // CHECK: fsub <2 x double>
  return _mm_sub_pd(A, B);
}

__m128d test_mm_sub_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_sub_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fsub double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_sub_sd(A, B);
}

__m128i test_mm_subs_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_subs_epi8
  // CHECK: call <16 x i8> @llvm.ssub.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_subs_epi8(A, B);
}

__m128i test_mm_subs_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_subs_epi16
  // CHECK: call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_subs_epi16(A, B);
}

__m128i test_mm_subs_epu8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_subs_epu8
  // CHECK-NOT: call <16 x i8> @llvm.x86.sse2.psubus.b(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: call <16 x i8> @llvm.usub.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_subs_epu8(A, B);
}

__m128i test_mm_subs_epu16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_subs_epu16
  // CHECK-NOT: call <8 x i16> @llvm.x86.sse2.psubus.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: call <8 x i16> @llvm.usub.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_subs_epu16(A, B);
}

int test_mm_ucomieq_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_ucomieq_sd
  // CHECK: call i32 @llvm.x86.sse2.ucomieq.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_ucomieq_sd(A, B);
}

int test_mm_ucomige_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_ucomige_sd
  // CHECK: call i32 @llvm.x86.sse2.ucomige.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_ucomige_sd(A, B);
}

int test_mm_ucomigt_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_ucomigt_sd
  // CHECK: call i32 @llvm.x86.sse2.ucomigt.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_ucomigt_sd(A, B);
}

int test_mm_ucomile_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_ucomile_sd
  // CHECK: call i32 @llvm.x86.sse2.ucomile.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_ucomile_sd(A, B);
}

int test_mm_ucomilt_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_ucomilt_sd
  // CHECK: call i32 @llvm.x86.sse2.ucomilt.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_ucomilt_sd(A, B);
}

int test_mm_ucomineq_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_ucomineq_sd
  // CHECK: call i32 @llvm.x86.sse2.ucomineq.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_ucomineq_sd(A, B);
}

__m128d test_mm_undefined_pd(void) {
  // X64-LABEL: test_mm_undefined_pd
  // X64: ret <2 x double> zeroinitializer
  //
  // X86-LABEL: test_mm_undefined_pd
  // X86: store <2 x double> zeroinitializer
  return _mm_undefined_pd();
}

__m128i test_mm_undefined_si128(void) {
  // CHECK-LABEL: test_mm_undefined_si128
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_undefined_si128();
}

__m128i test_mm_unpackhi_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_unpackhi_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  return _mm_unpackhi_epi8(A, B);
}

__m128i test_mm_unpackhi_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_unpackhi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  return _mm_unpackhi_epi16(A, B);
}

__m128i test_mm_unpackhi_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_unpackhi_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  return _mm_unpackhi_epi32(A, B);
}

__m128i test_mm_unpackhi_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_unpackhi_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 3>
  return _mm_unpackhi_epi64(A, B);
}

__m128d test_mm_unpackhi_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_unpackhi_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  return _mm_unpackhi_pd(A, B);
}

__m128i test_mm_unpacklo_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_unpacklo_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  return _mm_unpacklo_epi8(A, B);
}

__m128i test_mm_unpacklo_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_unpacklo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  return _mm_unpacklo_epi16(A, B);
}

__m128i test_mm_unpacklo_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_unpacklo_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  return _mm_unpacklo_epi32(A, B);
}

__m128i test_mm_unpacklo_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_unpacklo_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  return _mm_unpacklo_epi64(A, B);
}

__m128d test_mm_unpacklo_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_unpacklo_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>
  return _mm_unpacklo_pd(A, B);
}

__m128d test_mm_xor_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_xor_pd
  // CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  return _mm_xor_pd(A, B);
}

__m128i test_mm_xor_si128(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_xor_si128
  // CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  return _mm_xor_si128(A, B);
}

__m128d test_mm_cmp_pd_eq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_eq_oq
  // CHECK: fcmp oeq <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_EQ_OQ);
}

__m128d test_mm_cmp_pd_lt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_lt_os
  // CHECK: fcmp olt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_LT_OS);
}

__m128d test_mm_cmp_pd_le_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_le_os
  // CHECK: fcmp ole <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_LE_OS);
}

__m128d test_mm_cmp_pd_unord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_unord_q
  // CHECK: fcmp uno <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_UNORD_Q);
}

__m128d test_mm_cmp_pd_neq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_neq_uq
  // CHECK: fcmp une <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NEQ_UQ);
}

__m128d test_mm_cmp_pd_nlt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nlt_us
  // CHECK: fcmp uge <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NLT_US);
}

__m128d test_mm_cmp_pd_nle_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_nle_us
  // CHECK: fcmp ugt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_NLE_US);
}

__m128d test_mm_cmp_pd_ord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_ord_q
  // CHECK: fcmp ord <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd(a, b, _CMP_ORD_Q);
}

__m128d test_mm_cmp_sd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmp_sd
  // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 7)
  return _mm_cmp_sd(A, B, _CMP_ORD_Q);
}
