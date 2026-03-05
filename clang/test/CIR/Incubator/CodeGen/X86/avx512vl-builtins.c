// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s


#include <immintrin.h>

void test_mm_mask_storeu_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_storeu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 2>, !cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>)

  // LLVM-LABEL: @test_mm_mask_storeu_epi64
  // LLVM: call void @llvm.masked.store.v2i64.p0(<2 x i64> %{{.*}}, ptr elementtype(<2 x i64>) align 1 %{{.*}}, <2 x i1> %{{.*}})
  return _mm_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm_mask_storeu_epi32(void *__P, __mmask8 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_storeu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>)

  // LLVM-LABEL: @test_mm_mask_storeu_epi32
  // LLVM: call void @llvm.masked.store.v4i32.p0(<4 x i32> %{{.*}}, ptr elementtype(<4 x i32>) align 1 %{{.*}}, <4 x i1> %{{.*}})
  return _mm_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm_mask_storeu_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CIR-LABEL: _mm_mask_storeu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>)

  // LLVM-LABEL: @test_mm_mask_storeu_pd
  // LLVM: call void @llvm.masked.store.v2f64.p0(<2 x double> %{{.*}}, ptr elementtype(<2 x double>) align 1 %{{.*}}, <2 x i1> %{{.*}})
  return _mm_mask_storeu_pd(__P, __U, __A); 
}

void test_mm_mask_storeu_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CIR-LABEL: _mm_mask_storeu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 4>, !cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>)

  // LLVM-LABEL: @test_mm_mask_storeu_ps
  // LLVM: call void @llvm.masked.store.v4f32.p0(<4 x float> %{{.*}}, ptr elementtype(<4 x float>) align 1 %{{.*}}, <4 x i1> %{{.*}})
  return _mm_mask_storeu_ps(__P, __U, __A); 
}

void test_mm256_mask_storeu_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_storeu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 8>, !cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>)

  // LLVM-LABEL: @test_mm256_mask_storeu_epi32
  // LLVM: call void @llvm.masked.store.v8i32.p0(<8 x i32> %{{.*}}, ptr elementtype(<8 x i32>) align 1 %{{.*}}, <8 x i1> %{{.*}})
  return _mm256_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm256_mask_storeu_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_storeu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 4>, !cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>)

  // LLVM-LABEL: @test_mm256_mask_storeu_epi64
  // LLVM: call void @llvm.masked.store.v4i64.p0(<4 x i64> %{{.*}}, ptr elementtype(<4 x i64>) align 1 %{{.*}}, <4 x i1> %{{.*}})
  return _mm256_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm256_mask_storeu_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CIR-LABEL: _mm256_mask_storeu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 8>, !cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm256_mask_storeu_ps
  // LLVM: call void @llvm.masked.store.v8f32.p0(<8 x float> %{{.*}}, ptr elementtype(<8 x float>) align 1 %{{.*}}, <8 x i1> %{{.*}})
  return _mm256_mask_storeu_ps(__P, __U, __A); 
}

void test_mm_mask_store_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_store_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 2>, !cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_epi64
  // LLVM: call void @llvm.masked.store.v2i64.p0(<2 x i64> %{{.*}}, ptr elementtype(<2 x i64>) align 16 %{{.*}}, <2 x i1> %{{.*}})
  return _mm_mask_store_epi64(__P, __U, __A); 
}

void test_mm_mask_store_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CIR-LABEL: _mm_mask_store_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 4>, !cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_ps
  // LLVM: call void @llvm.masked.store.v4f32.p0(<4 x float> %{{.*}}, ptr elementtype(<4 x float>) align 16 %{{.*}}, <4 x i1> %{{.*}})
  return _mm_mask_store_ps(__P, __U, __A); 
}

void test_mm_mask_store_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CIR-LABEL: _mm_mask_store_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_pd
  // LLVM: call void @llvm.masked.store.v2f64.p0(<2 x double> %{{.*}}, ptr elementtype(<2 x double>) align 16 %{{.*}}, <2 x i1> %{{.*}})
  return _mm_mask_store_pd(__P, __U, __A); 
}

void test_mm256_mask_store_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_store_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 8>, !cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm256_mask_store_epi32
  // LLVM: call void @llvm.masked.store.v8i32.p0(<8 x i32> %{{.*}}, ptr elementtype(<8 x i32>) align 32 %{{.*}}, <8 x i1> %{{.*}})
  return _mm256_mask_store_epi32(__P, __U, __A); 
}

void test_mm256_mask_store_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_store_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 4>, !cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm256_mask_store_epi64
  // LLVM: call void @llvm.masked.store.v4i64.p0(<4 x i64> %{{.*}}, ptr elementtype(<4 x i64>) align 32 %{{.*}}, <4 x i1> %{{.*}})
  return _mm256_mask_store_epi64(__P, __U, __A); 
}

void test_mm256_mask_store_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CIR-LABEL: _mm256_mask_store_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 8>, !cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm256_mask_store_ps
  // LLVM: call void @llvm.masked.store.v8f32.p0(<8 x float> %{{.*}}, ptr elementtype(<8 x float>) align 32 %{{.*}}, <8 x i1> %{{.*}})
  return _mm256_mask_store_ps(__P, __U, __A); 
}

void test_mm256_mask_store_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CIR-LABEL: _mm256_mask_store_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 4>, !cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm256_mask_store_pd
  // LLVM: call void @llvm.masked.store.v4f64.p0(<4 x double> %{{.*}}, ptr elementtype(<4 x double>) align 32 %{{.*}}, <4 x i1> %{{.*}})
  return _mm256_mask_store_pd(__P, __U, __A); 
}
  
__m128 test_mm_mask_loadu_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_loadu_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_mask_loadu_ps
  // LLVM: @llvm.masked.load.v4f32.p0(ptr elementtype(<4 x float>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_loadu_ps(__W, __U, __P); 
}

__m128 test_mm_maskz_loadu_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_loadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_maskz_loadu_ps
  // LLVM: @llvm.masked.load.v4f32.p0(ptr elementtype(<4 x float>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_loadu_ps(__U, __P); 
}

__m256 test_mm256_mask_loadu_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_loadu_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_mask_loadu_ps
  // LLVM: @llvm.masked.load.v8f32.p0(ptr elementtype(<8 x float>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_loadu_ps(__W, __U, __P); 
}

__m256 test_mm256_maskz_loadu_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_loadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_maskz_loadu_ps
  // LLVM: @llvm.masked.load.v8f32.p0(ptr elementtype(<8 x float>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_loadu_ps(__U, __P); 
}

__m256d test_mm256_mask_loadu_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_loadu_pd
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_mask_loadu_pd
  // LLVM: @llvm.masked.load.v4f64.p0(ptr elementtype(<4 x double>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_loadu_pd(__W, __U, __P); 
}

__m128i test_mm_mask_loadu_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_loadu_epi32
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_mask_loadu_epi32
  // LLVM: @llvm.masked.load.v4i32.p0(ptr elementtype(<4 x i32>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_loadu_epi32(__W, __U, __P); 
}

__m256i test_mm256_mask_loadu_epi32(__m256i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_loadu_epi32
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_mask_loadu_epi32
  // LLVM: @llvm.masked.load.v8i32.p0(ptr elementtype(<8 x i32>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_mask_loadu_epi32(__W, __U, __P); 
}

__m128i test_mm_mask_loadu_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_loadu_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_mask_loadu_epi64
  // LLVM: @llvm.masked.load.v2i64.p0(ptr elementtype(<2 x i64>) align 1 %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_loadu_epi64(__W, __U, __P); 
}

__m256i test_mm256_mask_loadu_epi64(__m256i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_loadu_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s64i x 4>) -> !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_mask_loadu_epi64
  // LLVM: @llvm.masked.load.v4i64.p0(ptr elementtype(<4 x i64>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_mask_loadu_epi64(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_loadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s64i x 4>) -> !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_maskz_loadu_epi64
  // LLVM: @llvm.masked.load.v4i64.p0(ptr elementtype(<4 x i64>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskz_loadu_epi64(__U, __P); 
}

__m128 test_mm_mask_load_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_load_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_mask_load_ps
  // LLVM: @llvm.masked.load.v4f32.p0(ptr elementtype(<4 x float>) align 16 %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_load_ps(__W, __U, __P); 
}

__m128 test_mm_maskz_load_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_load_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_maskz_load_ps
  // LLVM: @llvm.masked.load.v4f32.p0(ptr elementtype(<4 x float>) align 16 %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_load_ps(__U, __P); 
}

__m256 test_mm256_mask_load_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_load_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_mask_load_ps
  // LLVM: @llvm.masked.load.v8f32.p0(ptr elementtype(<8 x float>) align 32 %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_load_ps(__W, __U, __P); 
}

__m256 test_mm256_maskz_load_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_load_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_maskz_load_ps
  // LLVM: @llvm.masked.load.v8f32.p0(ptr elementtype(<8 x float>) align 32 %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_load_ps(__U, __P); 
}

__m128d test_mm_mask_load_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_load_pd
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_mask_load_pd
  // LLVM: @llvm.masked.load.v2f64.p0(ptr elementtype(<2 x double>) align 16 %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_load_pd(__W, __U, __P); 
}

__m128d test_mm_maskz_load_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_load_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_maskz_load_pd
  // LLVM: @llvm.masked.load.v2f64.p0(ptr elementtype(<2 x double>) align 16 %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_load_pd(__U, __P); 
}

__m128d test_mm_maskz_loadu_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_loadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_maskz_loadu_pd
  // LLVM: @llvm.masked.load.v2f64.p0(ptr elementtype(<2 x double>) align 1 %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_loadu_pd(__U, __P); 
}

__m256d test_mm256_mask_load_pd(__m256d __W, __mmask8 __U, void const *__P) {
  //CIR-LABEL: _mm256_mask_load_pd
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_mask_load_pd
  // LLVM: @llvm.masked.load.v4f64.p0(ptr elementtype(<4 x double>) align 32 %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_load_pd(__W, __U, __P); 
}

__m256d test_mm256_maskz_load_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_load_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_maskz_load_pd
  // LLVM: @llvm.masked.load.v4f64.p0(ptr elementtype(<4 x double>) align 32 %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_load_pd(__U, __P); 
}

__m256d test_mm256_maskz_loadu_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_loadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_maskz_loadu_pd
  // LLVM: @llvm.masked.load.v4f64.p0(ptr elementtype(<4 x double>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_loadu_pd(__U, __P); 
}

__m128i test_mm_mask_load_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_load_epi32
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_mask_load_epi32
  // LLVM: @llvm.masked.load.v4i32.p0(ptr elementtype(<4 x i32>) align 16 %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_load_epi32(__W, __U, __P); 
}

__m128i test_mm_maskz_load_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_load_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_maskz_load_epi32
  // LLVM: @llvm.masked.load.v4i32.p0(ptr elementtype(<4 x i32>) align 16 %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_load_epi32(__U, __P); 
}

__m128i test_mm_maskz_loadu_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_loadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_maskz_loadu_epi32
  // LLVM: @llvm.masked.load.v4i32.p0(ptr elementtype(<4 x i32>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_loadu_epi32(__U, __P); 
}

__m256i test_mm256_maskz_load_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_load_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_maskz_load_epi32
  // LLVM: @llvm.masked.load.v8i32.p0(ptr elementtype(<8 x i32>) align 32 %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_load_epi32(__U, __P); 
}

__m256i test_mm256_maskz_loadu_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_loadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_maskz_loadu_epi32
  // LLVM: @llvm.masked.load.v8i32.p0(ptr elementtype(<8 x i32>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_loadu_epi32(__U, __P); 
}

__m128i test_mm_mask_load_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_load_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_mask_load_epi64
  // LLVM: @llvm.masked.load.v2i64.p0(ptr elementtype(<2 x i64>) align 16 %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_load_epi64(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_loadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_maskz_loadu_epi64
  // LLVM: @llvm.masked.load.v2i64.p0(ptr elementtype(<2 x i64>) align 1 %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_loadu_epi64(__U, __P); 
}

__m128i test_mm_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_load_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_maskz_load_epi64
  // LLVM: @llvm.masked.load.v2i64.p0(ptr elementtype(<2 x i64>) align 16 %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_load_epi64(__U, __P); 
}

__m256i test_mm256_mask_load_epi64(__m256i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_load_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s64i x 4>) -> !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_mask_load_epi64
  // LLVM: @llvm.masked.load.v4i64.p0(ptr elementtype(<4 x i64>) align 32 %{{.*}}, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_mask_load_epi64(__W, __U, __P); 
}

__m256i test_mm256_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_load_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s64i x 4>) -> !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_maskz_load_epi64
  // LLVM: @llvm.masked.load.v4i64.p0(ptr elementtype(<4 x i64>) align 32 %{{.*}}, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskz_load_epi64(__U, __P); 
}

__m128d test_mm_mask_expandloadu_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_expandloadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_mask_expandloadu_pd
  // LLVM: @llvm.masked.expandload.v2f64(ptr %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_expandloadu_pd(__W,__U,__P); 
}

__m128d test_mm_maskz_expandloadu_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_expandloadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %4, %8, %5 : (!cir.ptr<!cir.vector<!cir.double x 2>>, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_maskz_expandloadu_pd
  // LLVM: @llvm.masked.expandload.v2f64(ptr %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_expandloadu_pd(__U,__P); 
}

__m256d test_mm256_mask_expandloadu_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_expandloadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_mask_expandloadu_pd
  // LLVM: @llvm.masked.expandload.v4f64(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_expandloadu_pd(__W,__U,__P); 
}

__m256d test_mm256_maskz_expandloadu_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_expandloadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_maskz_expandloadu_pd
  // LLVM: @llvm.masked.expandload.v4f64(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_expandloadu_pd(__U,__P); 
}

__m128 test_mm_mask_expandloadu_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_expandloadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_mask_expandloadu_ps
  // LLVM: @llvm.masked.expandload.v4f32(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_expandloadu_ps(__W,__U,__P); 
}

__m128 test_mm_maskz_expandloadu_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_expandloadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_maskz_expandloadu_ps
  // LLVM: @llvm.masked.expandload.v4f32(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_expandloadu_ps(__U,__P); 
}

__m256 test_mm256_mask_expandloadu_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_expandloadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_mask_expandloadu_ps
  // LLVM: @llvm.masked.expandload.v8f32(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_expandloadu_ps(__W,__U,__P); 
}

__m256 test_mm256_maskz_expandloadu_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_expandloadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_maskz_expandloadu_ps
  // LLVM: @llvm.masked.expandload.v8f32(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_expandloadu_ps(__U,__P); 
}

__m128i test_mm_mask_expandloadu_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_expandloadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_mask_expandloadu_epi64
  // LLVM: @llvm.masked.expandload.v2i64(ptr %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_expandloadu_epi64(__W,__U,__P); 
}

__m128i test_mm_maskz_expandloadu_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_expandloadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_maskz_expandloadu_epi64
  // LLVM: @llvm.masked.expandload.v2i64(ptr %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_expandloadu_epi64(__U,__P); 
}

__m128i test_mm_mask_expandloadu_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_mask_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v4i32(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_expandloadu_epi32(__W,__U,__P); 
}

__m128i test_mm_maskz_expandloadu_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_maskz_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v4i32(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_expandloadu_epi32(__U,__P); 
}

__m256i test_mm256_mask_expandloadu_epi32(__m256i __W, __mmask8 __U,   void const *__P) {
  // CIR-LABEL: _mm256_mask_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_mask_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v8i32(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_mask_expandloadu_epi32(__W,__U,__P); 
}

__m256i test_mm256_maskz_expandloadu_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_maskz_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v8i32(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_expandloadu_epi32(__U,__P);
}

void test_mm_mask_compressstoreu_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CIR-LABEL: _mm_mask_compressstoreu_pd
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>, !cir.vector<!cir.int<s, 1> x 2>) -> !void

  // LLVM-LABEL: @test_mm_mask_compressstoreu_pd
  // LLVM: @llvm.masked.compressstore.v2f64(<2 x double> %{{.*}}, ptr %{{.*}}, <2 x i1> %{{.*}})
  return _mm_mask_compressstoreu_pd(__P,__U,__A); 
}

void test_mm256_mask_compressstoreu_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CIR-LABEL: _mm256_mask_compressstoreu_pd
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 4>, !cir.ptr<!cir.vector<!cir.double x 4>>, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm256_mask_compressstoreu_pd
  // LLVM: @llvm.masked.compressstore.v4f64(<4 x double> %{{.*}}, ptr %{{.*}}, <4 x i1> %{{.*}})
  return _mm256_mask_compressstoreu_pd(__P,__U,__A); 
}
void test_mm_mask_compressstoreu_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CIR-LABEL: _mm_mask_compressstoreu_ps
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 4>, !cir.ptr<!cir.vector<!cir.float x 4>>, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm_mask_compressstoreu_ps
  // LLVM: @llvm.masked.compressstore.v4f32(<4 x float> %{{.*}}, ptr %{{.*}}, <4 x i1> %{{.*}})
  return _mm_mask_compressstoreu_ps(__P,__U,__A); 
}

void test_mm256_mask_compressstoreu_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CIR-LABEL: _mm256_mask_compressstoreu_ps
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 8>, !cir.ptr<!cir.vector<!cir.float x 8>>, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm256_mask_compressstoreu_ps
  // LLVM: @llvm.masked.compressstore.v8f32(<8 x float> %{{.*}}, ptr %{{.*}}, <8 x i1> %{{.*}})
  return _mm256_mask_compressstoreu_ps(__P,__U,__A); 
}

void test_mm_mask_compressstoreu_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_compressstoreu_epi64
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 2>, !cir.ptr<!cir.vector<!s64i x 2>>, !cir.vector<!cir.int<s, 1> x 2>) -> !void

  // LLVM-LABEL: @test_mm_mask_compressstoreu_epi64
  // LLVM: @llvm.masked.compressstore.v2i64(<2 x i64> %{{.*}}, ptr %{{.*}}, <2 x i1> %{{.*}})
  return _mm_mask_compressstoreu_epi64(__P,__U,__A); 
}

void test_mm256_mask_compressstoreu_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_compressstoreu_epi64
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 4>, !cir.ptr<!cir.vector<!s64i x 4>>, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm256_mask_compressstoreu_epi64
  // LLVM: @llvm.masked.compressstore.v4i64(<4 x i64> %{{.*}}, ptr %{{.*}}, <4 x i1> %{{.*}})
  return _mm256_mask_compressstoreu_epi64(__P,__U,__A); 
}

void test_mm_mask_compressstoreu_epi32(void *__P, __mmask8 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_compressstoreu_epi32
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm_mask_compressstoreu_epi32
  // LLVM: @llvm.masked.compressstore.v4i32(<4 x i32> %{{.*}}, ptr %{{.*}}, <4 x i1> %{{.*}})
  return _mm_mask_compressstoreu_epi32(__P,__U,__A); 
}

void test_mm256_mask_compressstoreu_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_compressstoreu_epi32
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 8>, !cir.ptr<!cir.vector<!s32i x 8>>, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm256_mask_compressstoreu_epi32
  // LLVM: @llvm.masked.compressstore.v8i32(<8 x i32> %{{.*}}, ptr %{{.*}}, <8 x i1> %{{.*}})
  return _mm256_mask_compressstoreu_epi32(__P,__U,__A); 
}
__m128d test_mm_mmask_i64gather_pd(__m128d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mmask_i64gather_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3div2.df"

  // LLVM-LABEL: @test_mm_mmask_i64gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather3div2.df
  return _mm_mmask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mmask_i64gather_epi64(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mmask_i64gather_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3div2.di"

  // LLVM-LABEL: @test_mm_mmask_i64gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather3div2.di
  return _mm_mmask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_mmask_i64gather_pd(__m256d __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mmask_i64gather_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3div4.df"

  // LLVM-LABEL: @test_mm256_mmask_i64gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather3div4.df
  return _mm256_mmask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mmask_i64gather_epi64(__m256i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mmask_i64gather_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3div4.di"

  // LLVM-LABEL: @test_mm256_mmask_i64gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather3div4.di
  return _mm256_mmask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm_mmask_i64gather_ps(__m128 __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mmask_i64gather_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3div4.sf"

  // LLVM-LABEL: @test_mm_mmask_i64gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather3div4.sf
  return _mm_mmask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mmask_i64gather_epi32(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mmask_i64gather_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3div4.si"

  // LLVM-LABEL: @test_mm_mmask_i64gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather3div4.si
  return _mm_mmask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm256_mmask_i64gather_ps(__m128 __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mmask_i64gather_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3div8.sf"

  // LLVM-LABEL: @test_mm256_mmask_i64gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather3div8.sf
  return _mm256_mmask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm256_mmask_i64gather_epi32(__m128i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mmask_i64gather_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3div8.si"

  // LLVM-LABEL: @test_mm256_mmask_i64gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather3div8.si
  return _mm256_mmask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m128d test_mm_mask_i32gather_pd(__m128d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mask_i32gather_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3siv2.df"

  // LLVM-LABEL: @test_mm_mask_i32gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather3siv2.df
  return _mm_mmask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mask_i32gather_epi64(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mask_i32gather_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3siv2.di"

  // LLVM-LABEL: @test_mm_mask_i32gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather3siv2.di
  return _mm_mmask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_mask_i32gather_pd(__m256d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mask_i32gather_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3siv4.df"

  // LLVM-LABEL: @test_mm256_mask_i32gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather3siv4.df
  return _mm256_mmask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mask_i32gather_epi64(__m256i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mask_i32gather_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3siv4.di"

  // LLVM-LABEL: @test_mm256_mask_i32gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather3siv4.di
  return _mm256_mmask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm_mask_i32gather_ps(__m128 __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mask_i32gather_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3siv4.sf"

  // LLVM-LABEL: @test_mm_mask_i32gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather3siv4.sf
  return _mm_mmask_i32gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mask_i32gather_epi32(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mask_i32gather_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3siv4.si"

  // LLVM-LABEL: @test_mm_mask_i32gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather3siv4.si
  return _mm_mmask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m256 test_mm256_mask_i32gather_ps(__m256 __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mask_i32gather_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3siv8.sf"

  // LLVM-LABEL: @test_mm256_mask_i32gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather3siv8.sf
  return _mm256_mmask_i32gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mask_i32gather_epi32(__m256i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mask_i32gather_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather3siv8.si"

  // LLVM-LABEL: @test_mm256_mask_i32gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather3siv8.si
  return _mm256_mmask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

void test_mm_i64scatter_pd(double *__addr, __m128i __index,  __m128d __v1) {
  // CIR-LABEL: test_mm_i64scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv2.df"

  // LLVM-LABEL: @test_mm_i64scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatterdiv2.df
  return _mm_i64scatter_pd(__addr,__index,__v1,2); 
}

void test_mm_mask_i64scatter_pd(double *__addr, __mmask8 __mask, __m128i __index, __m128d __v1) {
  // CIR-LABEL: test_mm_mask_i64scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv2.df"

  // LLVM-LABEL: @test_mm_mask_i64scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatterdiv2.df
  return _mm_mask_i64scatter_pd(__addr,__mask,__index,__v1,2); 
}

void test_mm_i64scatter_epi64(long long *__addr, __m128i __index,  __m128i __v1) {
  // CIR-LABEL: test_mm_i64scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv2.di"

  // LLVM-LABEL: @test_mm_i64scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scatterdiv2.di
  return _mm_i64scatter_epi64(__addr,__index,__v1,2); 
}

void test_mm_mask_i64scatter_epi64(long long *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CIR-LABEL: test_mm_mask_i64scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv2.di"

  // LLVM-LABEL: @test_mm_mask_i64scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scatterdiv2.di
  return _mm_mask_i64scatter_epi64(__addr,__mask,__index,__v1,2); 
}

void test_mm256_i64scatter_pd(double *__addr, __m256i __index,  __m256d __v1) {
  // CIR-LABEL: test_mm256_i64scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv4.df"

  // LLVM-LABEL: @test_mm256_i64scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatterdiv4.df
  return _mm256_i64scatter_pd(__addr,__index,__v1,2); 
}

void test_mm256_mask_i64scatter_pd(double *__addr, __mmask8 __mask, __m256i __index, __m256d __v1) {
  // CIR-LABEL: test_mm256_mask_i64scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv4.df"

  // LLVM-LABEL: @test_mm256_mask_i64scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatterdiv4.df
  return _mm256_mask_i64scatter_pd(__addr,__mask,__index,__v1,2); 
}

void test_mm256_i64scatter_epi64(long long *__addr, __m256i __index,  __m256i __v1) {
  // CIR-LABEL: test_mm256_i64scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv4.di"

  // LLVM-LABEL: @test_mm256_i64scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scatterdiv4.di
  return _mm256_i64scatter_epi64(__addr,__index,__v1,2); 
}

void test_mm256_mask_i64scatter_epi64(long long *__addr, __mmask8 __mask,  __m256i __index, __m256i __v1) {
  // CIR-LABEL: test_mm256_mask_i64scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv4.di"

  // LLVM-LABEL: @test_mm256_mask_i64scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scatterdiv4.di
  return _mm256_mask_i64scatter_epi64(__addr,__mask,__index,__v1,2); 
}

void test_mm_i64scatter_ps(float *__addr, __m128i __index, __m128 __v1) {
  // CIR-LABEL: test_mm_i64scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv4.sf"

  // LLVM-LABEL: @test_mm_i64scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatterdiv4.sf
  return _mm_i64scatter_ps(__addr,__index,__v1,2); 
}

void test_mm_mask_i64scatter_ps(float *__addr, __mmask8 __mask, __m128i __index, __m128 __v1) {
  // CIR-LABEL: test_mm_mask_i64scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv4.sf"

  // LLVM-LABEL: @test_mm_mask_i64scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatterdiv4.sf
  return _mm_mask_i64scatter_ps(__addr,__mask,__index,__v1,2); 
}

void test_mm_i64scatter_epi32(int *__addr, __m128i __index,  __m128i __v1) {
  // CIR-LABEL: test_mm_i64scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv4.si"

  // LLVM-LABEL: @test_mm_i64scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatterdiv4.si
  return _mm_i64scatter_epi32(__addr,__index,__v1,2); 
}

void test_mm_mask_i64scatter_epi32(int *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CIR-LABEL: test_mm_mask_i64scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv4.si"

  // LLVM-LABEL: @test_mm_mask_i64scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatterdiv4.si
  return _mm_mask_i64scatter_epi32(__addr,__mask,__index,__v1,2); 
}

void test_mm256_i64scatter_ps(float *__addr, __m256i __index,  __m128 __v1) {
  // CIR-LABEL: test_mm256_i64scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv8.sf"

  // LLVM-LABEL: @test_mm256_i64scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatterdiv8.sf
  return _mm256_i64scatter_ps(__addr,__index,__v1,2); 
}

void test_mm256_mask_i64scatter_ps(float *__addr, __mmask8 __mask, __m256i __index, __m128 __v1) {
  // CIR-LABEL: test_mm256_mask_i64scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv8.sf"

  // LLVM-LABEL: @test_mm256_mask_i64scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatterdiv8.sf
  return _mm256_mask_i64scatter_ps(__addr,__mask,__index,__v1,2); 
}

void test_mm256_i64scatter_epi32(int *__addr, __m256i __index,  __m128i __v1) {
  // CIR-LABEL: test_mm256_i64scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv8.si"

  // LLVM-LABEL: @test_mm256_i64scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatterdiv8.si
  return _mm256_i64scatter_epi32(__addr,__index,__v1,2); 
}

void test_mm256_mask_i64scatter_epi32(int *__addr, __mmask8 __mask,  __m256i __index, __m128i __v1) {
  // CIR-LABEL: test_mm256_mask_i64scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatterdiv8.si"

  // LLVM-LABEL: @test_mm256_mask_i64scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatterdiv8.si
  return _mm256_mask_i64scatter_epi32(__addr,__mask,__index,__v1,2); 
}

void test_mm_i32scatter_pd(double *__addr, __m128i __index,  __m128d __v1) {
  // CIR-LABEL: test_mm_i32scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv2.df"

  // LLVM-LABEL: @test_mm_i32scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scattersiv2.df
  return _mm_i32scatter_pd(__addr,__index,__v1,2); 
}

void test_mm_mask_i32scatter_pd(double *__addr, __mmask8 __mask, __m128i __index, __m128d __v1) {
  // CIR-LABEL: test_mm_mask_i32scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv2.df"

  // LLVM-LABEL: @test_mm_mask_i32scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scattersiv2.df
  return _mm_mask_i32scatter_pd(__addr,__mask,__index,__v1,2); 
}

void test_mm_i32scatter_epi64(long long *__addr, __m128i __index,  __m128i __v1) {
  // CIR-LABEL: test_mm_i32scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv2.di"

  // LLVM-LABEL: @test_mm_i32scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scattersiv2.di
  return _mm_i32scatter_epi64(__addr,__index,__v1,2); 
}

void test_mm_mask_i32scatter_epi64(long long *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CIR-LABEL: test_mm_mask_i32scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv2.di"

  // LLVM-LABEL: @test_mm_mask_i32scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scattersiv2.di
  return _mm_mask_i32scatter_epi64(__addr,__mask,__index,__v1,2); 
}

void test_mm256_i32scatter_pd(double *__addr, __m128i __index,  __m256d __v1) {
  // CIR-LABEL: test_mm256_i32scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv4.df"

  // LLVM-LABEL: @test_mm256_i32scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scattersiv4.df
  return _mm256_i32scatter_pd(__addr,__index,__v1,2); 
}

void test_mm256_mask_i32scatter_pd(double *__addr, __mmask8 __mask, __m128i __index, __m256d __v1) {
  // CIR-LABEL: test_mm256_mask_i32scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv4.df"

  // LLVM-LABEL: @test_mm256_mask_i32scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scattersiv4.df
  return _mm256_mask_i32scatter_pd(__addr,__mask,__index,__v1,2); 
}

void test_mm256_i32scatter_epi64(long long *__addr, __m128i __index,  __m256i __v1) {
  // CIR-LABEL: test_mm256_i32scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv4.di"

  // LLVM-LABEL: @test_mm256_i32scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scattersiv4.di
  return _mm256_i32scatter_epi64(__addr,__index,__v1,2); 
}

void test_mm256_mask_i32scatter_epi64(long long *__addr, __mmask8 __mask,  __m128i __index, __m256i __v1) {
  // CIR-LABEL: test_mm256_mask_i32scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv4.di"

  // LLVM-LABEL: @test_mm256_mask_i32scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scattersiv4.di
  return _mm256_mask_i32scatter_epi64(__addr,__mask,__index,__v1,2); 
}

void test_mm_i32scatter_ps(float *__addr, __m128i __index, __m128 __v1) {
  // CIR-LABEL: test_mm_i32scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv4.sf"

  // LLVM-LABEL: @test_mm_i32scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scattersiv4.sf
  return _mm_i32scatter_ps(__addr,__index,__v1,2); 
}

void test_mm_mask_i32scatter_ps(float *__addr, __mmask8 __mask, __m128i __index, __m128 __v1) {
  // CIR-LABEL: test_mm_mask_i32scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv4.sf"

  // LLVM-LABEL: @test_mm_mask_i32scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scattersiv4.sf
  return _mm_mask_i32scatter_ps(__addr,__mask,__index,__v1,2); 
}

void test_mm_i32scatter_epi32(int *__addr, __m128i __index,  __m128i __v1) {
  // CIR-LABEL: test_mm_i32scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv4.si"

  // LLVM-LABEL: @test_mm_i32scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scattersiv4.si
  return _mm_i32scatter_epi32(__addr,__index,__v1,2); 
}

void test_mm_mask_i32scatter_epi32(int *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CIR-LABEL: test_mm_mask_i32scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv4.si"

  // LLVM-LABEL: @test_mm_mask_i32scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scattersiv4.si
  return _mm_mask_i32scatter_epi32(__addr,__mask,__index,__v1,2); 
}

void test_mm256_i32scatter_ps(float *__addr, __m256i __index,  __m256 __v1) {
  // CIR-LABEL: test_mm256_i32scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv8.sf"

  // LLVM-LABEL: @test_mm256_i32scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scattersiv8.sf
  return _mm256_i32scatter_ps(__addr,__index,__v1,2); 
}

void test_mm256_mask_i32scatter_ps(float *__addr, __mmask8 __mask, __m256i __index, __m256 __v1) {
  // CIR-LABEL: test_mm256_mask_i32scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv8.sf"

  // LLVM-LABEL: @test_mm256_mask_i32scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scattersiv8.sf
  return _mm256_mask_i32scatter_ps(__addr,__mask,__index,__v1,2); 
}

void test_mm256_i32scatter_epi32(int *__addr, __m256i __index,  __m256i __v1) {
  // CIR-LABEL: test_mm256_i32scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv8.si"

  // LLVM-LABEL: @test_mm256_i32scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scattersiv8.si
  return _mm256_i32scatter_epi32(__addr,__index,__v1,2); 
}

void test_mm256_mask_i32scatter_epi32(int *__addr, __mmask8 __mask,  __m256i __index, __m256i __v1) {
  // CIR-LABEL: test_mm256_mask_i32scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scattersiv8.si"

  // LLVM-LABEL: @test_mm256_mask_i32scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scattersiv8.si
  return _mm256_mask_i32scatter_epi32(__addr,__mask,__index,__v1,2); 
}

__m256 test_mm256_insertf32x4(__m256 __A, __m128 __B) {
  // CIR-LABEL: test_mm256_insertf32x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_insertf32x4
  // LLVM: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_insertf32x4(__A, __B, 1); 
}

__m256i test_mm256_inserti32x4(__m256i __A, __m128i __B) {
  // CIR-LABEL: test_mm256_inserti32x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s32i x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<!s32i x 8> 

  // LLVM-LABEL: @test_mm256_inserti32x4
  // LLVM: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_inserti32x4(__A, __B, 1); 
}
