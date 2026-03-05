// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m512 test_mm512_undefined(void) {
  // CIR-LABEL: _mm512_undefined
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 8>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 8> -> !cir.vector<!cir.float x 16>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.float x 16>

  // LLVM-LABEL: test_mm512_undefined
  // LLVM: store <16 x float> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <16 x float>, ptr %[[A]], align 64
  // LLVM: ret <16 x float> %{{.*}}
  return _mm512_undefined();
}

__m512 test_mm512_undefined_ps(void) {
  // CIR-LABEL: _mm512_undefined_ps
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 8>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 8> -> !cir.vector<!cir.float x 16>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.float x 16>

  // LLVM-LABEL: test_mm512_undefined_ps
  // LLVM: store <16 x float> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <16 x float>, ptr %[[A]], align 64
  // LLVM: ret <16 x float> %{{.*}}
  return _mm512_undefined_ps();
}

__m512d test_mm512_undefined_pd(void) {
  // CIR-LABEL: _mm512_undefined_pd
  // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<!cir.double x 8>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.double x 8>

  // LLVM-LABEL: test_mm512_undefined_pd
  // LLVM: store <8 x double> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <8 x double>, ptr %[[A]], align 64
  // LLVM: ret <8 x double> %{{.*}}
  return _mm512_undefined_pd();
}

__m512i test_mm512_undefined_epi32(void) {
  // CIR-LABEL: _mm512_undefined_epi32
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 8>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 8> -> !cir.vector<!s64i x 8>
  // CIR: cir.return %{{.*}} : !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_undefined_epi32
  // LLVM: store <8 x i64> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <8 x i64>, ptr %[[A]], align 64
  // LLVM: ret <8 x i64> %{{.*}}
  return _mm512_undefined_epi32();
}

void test_mm512_mask_storeu_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_storeu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 8>, !cir.ptr<!s64i>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: test_mm512_mask_storeu_epi64
  // LLVM: call void @llvm.masked.store.v8i64.p0(<8 x i64> %{{.*}}, ptr elementtype(<8 x i64>) align 1 %{{.*}}, <8 x i1> %{{.*}})
  return _mm512_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm512_mask_storeu_epi32(void *__P, __mmask16 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_storeu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 16>, !cir.ptr<!s32i>, !u32i, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: test_mm512_mask_storeu_epi32
  // LLVM: call void @llvm.masked.store.v16i32.p0(<16 x i32> %{{.*}}, ptr elementtype(<16 x i32>) align 1 %{{.*}}, <16 x i1> %{{.*}})
  return _mm512_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm_mask_store_ss(float * __P, __mmask8 __U, __m128 __A){
  // CIR-LABEL: _mm_mask_store_ss
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 4>, !cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: test_mm_mask_store_ss
  // LLVM: call void @llvm.masked.store.v4f32.p0(<4 x float> %{{.*}}, ptr elementtype(<4 x float>) align 1 %{{.*}}, <4 x i1> %{{.*}})

  _mm_mask_store_ss(__P, __U, __A);
}

void test_mm_mask_store_sd(double * __P, __mmask8 __U, __m128d __A){
  // CIR-LABEL: _mm_mask_store_sd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>) -> !void

  // LLVM-LABEL: test_mm_mask_store_sd
  // LLVM: call void @llvm.masked.store.v2f64.p0(<2 x double> %{{.*}}, ptr elementtype(<2 x double>) align 1 %{{.*}}, <2 x i1> %{{.*}})
  _mm_mask_store_sd(__P, __U, __A);
}

void test_mm512_mask_store_pd(void *p, __m512d a, __mmask8 m){
  // CIR-LABEL: _mm512_mask_store_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 8>, !cir.ptr<!cir.vector<!cir.double x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: test_mm512_mask_store_pd
  // LLVM: call void @llvm.masked.store.v8f64.p0(<8 x double> %{{.*}}, ptr elementtype(<8 x double>) align 64 %{{.*}}, <8 x i1> %{{.*}})
  _mm512_mask_store_pd(p, m, a);
}

void test_mm512_mask_store_epi32(void *__P, __mmask16 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_store_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 16>, !cir.ptr<!cir.vector<!s32i x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: test_mm512_mask_store_epi32
  // LLVM: call void @llvm.masked.store.v16i32.p0(<16 x i32> %{{.*}}, ptr elementtype(<16 x i32>) align 64 %{{.*}}, <16 x i1> %{{.*}})
  return _mm512_mask_store_epi32(__P, __U, __A); 
}

void test_mm512_mask_store_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_store_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 8>, !cir.ptr<!cir.vector<!s64i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: test_mm512_mask_store_epi64
  // LLVM: call void @llvm.masked.store.v8i64.p0(<8 x i64> %{{.*}}, ptr elementtype(<8 x i64>) align 64 %{{.*}}, <8 x i1> %{{.*}})
  return _mm512_mask_store_epi64(__P, __U, __A); 
}

void test_mm512_mask_store_ps(void *p, __m512 a, __mmask16 m){
  // CIR-LABEL: _mm512_mask_store_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 16>, !cir.ptr<!cir.vector<!cir.float x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: test_mm512_mask_store_ps
  // LLVM: call void @llvm.masked.store.v16f32.p0(<16 x float> %{{.*}}, ptr elementtype(<16 x float>) align 64 %{{.*}}, <16 x i1> %{{.*}})
  _mm512_mask_store_ps(p, m, a);
}

__m512 test_mm512_mask_loadu_ps (__m512 __W, __mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_loadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.float>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!cir.float x 16>) -> !cir.vector<!cir.float x 16>

  // LLVM-LABEL: test_mm512_mask_loadu_ps
  // LLVM: @llvm.masked.load.v16f32.p0(ptr elementtype(<16 x float>) align 1 %{{.*}}, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_mask_loadu_ps (__W,__U, __P);
}

__m512 test_mm512_maskz_load_ps(__mmask16 __U, void *__P)
{

  // CIR-LABEL: _mm512_maskz_load_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!cir.float x 16>) -> !cir.vector<!cir.float x 16>

  // LLVM-LABEL: test_mm512_maskz_load_ps
  // LLVM: @llvm.masked.load.v16f32.p0(ptr elementtype(<16 x float>) align 64 %{{.*}}, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_maskz_load_ps(__U, __P);
}

__m512d test_mm512_mask_loadu_pd (__m512d __W, __mmask8 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_loadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.double>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.double x 8>) -> !cir.vector<!cir.double x 8>

  // LLVM-LABEL: test_mm512_mask_loadu_pd
  // LLVM: @llvm.masked.load.v8f64.p0(ptr elementtype(<8 x double>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_mask_loadu_pd (__W,__U, __P);
}

__m512d test_mm512_maskz_load_pd(__mmask8 __U, void *__P)
{
  // CIR-LABEL: _mm512_maskz_load_pd
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.double x 8>) -> !cir.vector<!cir.double x 8>

  // LLVM-LABEL: test_mm512_maskz_load_pd
  // LLVM: @llvm.masked.load.v8f64.p0(ptr elementtype(<8 x double>) align 64 %{{.*}}, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_maskz_load_pd(__U, __P);
}

__m512i test_mm512_mask_loadu_epi32 (__m512i __W, __mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_loadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_mask_loadu_epi32
  // LLVM: @llvm.masked.load.v16i32.p0(ptr elementtype(<16 x i32>) align 1 %{{.*}}, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_mask_loadu_epi32 (__W,__U, __P);
}

__m512i test_mm512_maskz_loadu_epi32 (__mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_maskz_loadu_epi32
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_maskz_loadu_epi32
  // LLVM: @llvm.masked.load.v16i32.p0(ptr elementtype(<16 x i32>) align 1 %{{.*}}, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_maskz_loadu_epi32 (__U, __P);
}

__m512i test_mm512_mask_loadu_epi64 (__m512i __W, __mmask8 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_loadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_mask_loadu_epi64 
  // LLVM: @llvm.masked.load.v8i64.p0(ptr elementtype(<8 x i64>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_mask_loadu_epi64 (__W,__U, __P);
}

__m512i test_mm512_maskz_loadu_epi64 (__mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_maskz_loadu_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_maskz_loadu_epi64
  // LLVM: @llvm.masked.load.v8i64.p0(ptr elementtype(<8 x i64>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_maskz_loadu_epi64 (__U, __P);
}

__m128 test_mm_mask_load_ss(__m128 __A, __mmask8 __U, const float* __W)
{
  // CIR-LABEL: _mm_mask_load_ss
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>
  
  // LLVM-LABEL: test_mm_mask_load_ss
  // LLVM: call {{.*}}<4 x float> @llvm.masked.load.v4f32.p0(ptr elementtype(<4 x float>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_load_ss(__A, __U, __W);
}

__m128 test_mm_maskz_load_ss (__mmask8 __U, const float * __W)
{
  // CIR-LABEL: _mm_maskz_load_ss
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: test_mm_maskz_load_ss
  // LLVM: call {{.*}}<4 x float> @llvm.masked.load.v4f32.p0(ptr elementtype(<4 x float>) align 1 %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_load_ss (__U, __W);
}

__m128d test_mm_mask_load_sd (__m128d __A, __mmask8 __U, const double * __W)
{
  // CIR-LABEL: _mm_mask_load_sd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: test_mm_mask_load_sd
  // LLVM: call {{.*}}<2 x double> @llvm.masked.load.v2f64.p0(ptr elementtype(<2 x double>) align 1 %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_load_sd (__A, __U, __W);
}

__m128d test_mm_maskz_load_sd (__mmask8 __U, const double * __W)
{
  // CIR-LABEL: _mm_maskz_load_sd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: test_mm_maskz_load_sd
  // LLVM: call {{.*}}<2 x double> @llvm.masked.load.v2f64.p0(ptr elementtype(<2 x double>) align 1 %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_load_sd (__U, __W);
}

__m512 test_mm512_mask_load_ps (__m512 __W, __mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_load_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!cir.float x 16>) -> !cir.vector<!cir.float x 16>

  // LLVM-LABEL: test_mm512_mask_load_ps
  // LLVM: @llvm.masked.load.v16f32.p0(ptr elementtype(<16 x float>) align 64 %{{.*}}, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_mask_load_ps (__W,__U, __P);
}

__m512d test_mm512_mask_load_pd (__m512d __W, __mmask8 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_load_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.double x 8>) -> !cir.vector<!cir.double x 8>

  // LLVM-LABEL: test_mm512_mask_load_pd
  // LLVM: @llvm.masked.load.v8f64.p0(ptr elementtype(<8 x double>) align 64 %{{.*}}, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_mask_load_pd (__W,__U, __P);
}

__m512i test_mm512_mask_load_epi32(__m512i __W, __mmask16 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_load_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_mask_load_epi32
  // LLVM: @llvm.masked.load.v16i32.p0(ptr elementtype(<16 x i32>) align 64 %{{.*}}, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_mask_load_epi32(__W, __U, __P); 
}

__m512i test_mm512_mask_load_epi64(__m512i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_load_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_mask_load_epi64
  // LLVM: @llvm.masked.load.v8i64.p0(ptr elementtype(<8 x i64>) align 64 %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_mask_load_epi64(__W, __U, __P); 
}

__m512i test_mm512_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_load_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_maskz_load_epi64
  // LLVM: @llvm.masked.load.v8i64.p0(ptr elementtype(<8 x i64>) align 64 %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_maskz_load_epi64(__U, __P); 
}

__m512i test_mm512_mask_expandloadu_epi64(__m512i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_expandloadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_mask_expandloadu_epi64
  // LLVM: @llvm.masked.expandload.v8i64(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_mask_expandloadu_epi64(__W, __U, __P); 
}

__m512i test_mm512_maskz_expandloadu_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_expandloadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_maskz_expandloadu_epi64
  // LLVM: @llvm.masked.expandload.v8i64(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_maskz_expandloadu_epi64(__U, __P); 
}

__m512i test_mm512_mask_expandloadu_epi32(__m512i __W, __mmask16 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 16>>, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_mask_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v16i32(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_mask_expandloadu_epi32(__W, __U, __P); 
}

__m512i test_mm512_maskz_expandloadu_epi32(__mmask16 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 16>>, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_maskz_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v16i32(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_maskz_expandloadu_epi32(__U, __P); 
}

void test_mm512_mask_compressstoreu_pd(void *__P, __mmask8 __U, __m512d __A) {
  // CIR-LABEL: _mm512_mask_compressstoreu_pd
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 8>, !cir.ptr<!cir.vector<!cir.double x 8>>, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: test_mm512_mask_compressstoreu_pd
  // LLVM: @llvm.masked.compressstore.v8f64(<8 x double> %{{.*}}, ptr %{{.*}}, <8 x i1> %{{.*}})
  return _mm512_mask_compressstoreu_pd(__P, __U, __A); 
}

void test_mm512_mask_compressstoreu_ps(void *__P, __mmask16 __U, __m512 __A) {
  // CIR-LABEL: _mm512_mask_compressstoreu_ps
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 16>, !cir.ptr<!cir.vector<!cir.float x 16>>, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: test_mm512_mask_compressstoreu_ps
  // LLVM: @llvm.masked.compressstore.v16f32(<16 x float> %{{.*}}, ptr %{{.*}}, <16 x i1> %{{.*}})
  return _mm512_mask_compressstoreu_ps(__P, __U, __A); 
}

void test_mm512_mask_compressstoreu_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_compressstoreu_epi64
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 8>, !cir.ptr<!cir.vector<!s64i x 8>>, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: test_mm512_mask_compressstoreu_epi64
  // LLVM: @llvm.masked.compressstore.v8i64(<8 x i64> %{{.*}}, ptr %{{.*}}, <8 x i1> %{{.*}})
  return _mm512_mask_compressstoreu_epi64(__P, __U, __A); 
}

void test_mm512_mask_compressstoreu_epi32(void *__P, __mmask16 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_compressstoreu_epi32
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 16>, !cir.ptr<!cir.vector<!s32i x 16>>, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: test_mm512_mask_compressstoreu_epi32
  // LLVM: @llvm.masked.compressstore.v16i32(<16 x i32> %{{.*}}, ptr %{{.*}}, <16 x i1> %{{.*}})
  return _mm512_mask_compressstoreu_epi32(__P, __U, __A); 
}
__m512d test_mm512_i32gather_pd(__m256i __index, void const *__addr) {
  // CIR-LABEL: _mm512_i32gather_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.dpd.512"

  // LLVM-LABEL: test_mm512_i32gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_i32gather_pd(__index, __addr, 2); 
}

__m512d test_mm512_mask_i32gather_pd(__m512d __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: _mm512_mask_i32gather_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.dpd.512"

  // LLVM-LABEL: test_mm512_mask_i32gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_mask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m512 test_mm512_i32gather_ps(__m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_i32gather_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.dps.512"

  // LLVM-LABEL: test_mm512_i32gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather.dps.512
  return _mm512_i32gather_ps(__index, __addr, 2); 
}

__m512d test_mm512_i64gather_pd(__m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_i64gather_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.qpd.512"

  // LLVM-LABEL: test_mm512_i64gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather.qpd.512
  return _mm512_i64gather_pd(__index, __addr, 2); 
}

__m512d test_mm512_mask_i64gather_pd(__m512d __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_mask_i64gather_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.qpd.512"

  // LLVM-LABEL: test_mm512_mask_i64gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather.qpd.512
  return _mm512_mask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m256 test_mm512_i64gather_ps(__m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_i64gather_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.qps.512"

  // LLVM-LABEL: test_mm512_i64gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather.qps.512
  return _mm512_i64gather_ps(__index, __addr, 2); 
}

__m256 test_mm512_mask_i64gather_ps(__m256 __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_mask_i64gather_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.qps.512"

  // LLVM-LABEL: test_mm512_mask_i64gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather.qps.512
  return _mm512_mask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m512i test_mm512_i32gather_epi64(__m256i __index, void const *__addr) {
  // CIR-LABEL: _mm512_i32gather_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.dpq.512"

  // LLVM-LABEL: test_mm512_i32gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_i32gather_epi64(__index, __addr, 2); 
}

__m512i test_mm512_mask_i32gather_epi64(__m512i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: _mm512_mask_i32gather_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.dpq.512"

  // LLVM-LABEL: test_mm512_mask_i32gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_mask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m512i test_mm512_i32gather_epi32(__m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_i32gather_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.dpi.512"

  // LLVM-LABEL: test_mm512_i32gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather.dpi.512
  return _mm512_i32gather_epi32(__index, __addr, 2); 
}

__m512i test_mm512_mask_i32gather_epi32(__m512i __v1_old, __mmask16 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_mask_i32gather_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.dpi.512"

  // LLVM-LABEL: test_mm512_mask_i32gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather.dpi.512
  return _mm512_mask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m512i test_mm512_i64gather_epi64(__m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_i64gather_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.qpq.512"

  // LLVM-LABEL: test_mm512_i64gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather.qpq.512
  return _mm512_i64gather_epi64(__index, __addr, 2); 
}

__m512i test_mm512_mask_i64gather_epi64(__m512i __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_mask_i64gather_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.qpq.512"

  // LLVM-LABEL: test_mm512_mask_i64gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather.qpq.512
  return _mm512_mask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm512_i64gather_epi32(__m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_i64gather_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.qpi.512"

  // LLVM-LABEL: test_mm512_i64gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather.qpi.512
  return _mm512_i64gather_epi32(__index, __addr, 2); 
}

__m256i test_mm512_mask_i64gather_epi32(__m256i __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: _mm512_mask_i64gather_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.gather.qpi.512"

  // LLVM-LABEL: test_mm512_mask_i64gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather.qpi.512
  return _mm512_mask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}


void test_mm512_i32scatter_pd(void *__addr, __m256i __index, __m512d __v1) {
  // CIR-LABEL: test_mm512_i32scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.dpd.512"

  // LLVM-LABEL: test_mm512_i32scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatter.dpd.512
  return _mm512_i32scatter_pd(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i32scatter_pd(void *__addr, __mmask8 __mask, __m256i __index, __m512d __v1) {
  // CIR-LABEL: test_mm512_mask_i32scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.dpd.512"

  // LLVM-LABEL: test_mm512_mask_i32scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatter.dpd.512
  return _mm512_mask_i32scatter_pd(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i32scatter_ps(void *__addr, __m512i __index, __m512 __v1) {
  // CIR-LABEL: test_mm512_i32scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.dps.512"

  // LLVM-LABEL: test_mm512_i32scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatter.dps.512
  return _mm512_i32scatter_ps(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i32scatter_ps(void *__addr, __mmask16 __mask, __m512i __index, __m512 __v1) {
  // CIR-LABEL: test_mm512_mask_i32scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.dps.512"

  // LLVM-LABEL: test_mm512_mask_i32scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatter.dps.512
  return _mm512_mask_i32scatter_ps(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i64scatter_pd(void *__addr, __m512i __index, __m512d __v1) {
  // CIR-LABEL: test_mm512_i64scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.qpd.512"

  // LLVM-LABEL: test_mm512_i64scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatter.qpd.512
  return _mm512_i64scatter_pd(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i64scatter_pd(void *__addr, __mmask8 __mask, __m512i __index, __m512d __v1) {
  // CIR-LABEL: test_mm512_mask_i64scatter_pd
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.qpd.512"

  // LLVM-LABEL: test_mm512_mask_i64scatter_pd
  // LLVM: @llvm.x86.avx512.mask.scatter.qpd.512
  return _mm512_mask_i64scatter_pd(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i64scatter_ps(void *__addr, __m512i __index, __m256 __v1) {
  // CIR-LABEL: test_mm512_i64scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.qps.512"

  // LLVM-LABEL: test_mm512_i64scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatter.qps.512
  return _mm512_i64scatter_ps(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i64scatter_ps(void *__addr, __mmask8 __mask, __m512i __index, __m256 __v1) {
  // CIR-LABEL: test_mm512_mask_i64scatter_ps
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.qps.512"

  // LLVM-LABEL: test_mm512_mask_i64scatter_ps
  // LLVM: @llvm.x86.avx512.mask.scatter.qps.512
  return _mm512_mask_i64scatter_ps(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i32scatter_epi32(void *__addr, __m512i __index, __m512i __v1) {
  // CIR-LABEL: test_mm512_i32scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.dpi.512"

  // LLVM-LABEL: test_mm512_i32scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatter.dpi.512
  return _mm512_i32scatter_epi32(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i32scatter_epi32(void *__addr, __mmask16 __mask, __m512i __index, __m512i __v1) {
  // CIR-LABEL: test_mm512_mask_i32scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.dpi.512"

  // LLVM-LABEL: test_mm512_mask_i32scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatter.dpi.512
  return _mm512_mask_i32scatter_epi32(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i64scatter_epi64(void *__addr, __m512i __index, __m512i __v1) {
  // CIR-LABEL: test_mm512_i64scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.qpq.512"

  // LLVM-LABEL: test_mm512_i64scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scatter.qpq.512
  return _mm512_i64scatter_epi64(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i64scatter_epi64(void *__addr, __mmask8 __mask, __m512i __index, __m512i __v1) {
  // CIR-LABEL: test_mm512_mask_i64scatter_epi64
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.qpq.512"

  // LLVM-LABEL: test_mm512_mask_i64scatter_epi64
  // LLVM: @llvm.x86.avx512.mask.scatter.qpq.512
  return _mm512_mask_i64scatter_epi64(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i64scatter_epi32(void *__addr, __m512i __index, __m256i __v1) {
  // CIR-LABEL: test_mm512_i64scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.qpi.512"

  // LLVM-LABEL: test_mm512_i64scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatter.qpi.512
  return _mm512_i64scatter_epi32(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i64scatter_epi32(void *__addr, __mmask8 __mask, __m512i __index, __m256i __v1) {
  // CIR-LABEL: test_mm512_mask_i64scatter_epi32
  // CIR: cir.llvm.intrinsic "x86.avx512.mask.scatter.qpi.512"

  // LLVM-LABEL: test_mm512_mask_i64scatter_epi32
  // LLVM: @llvm.x86.avx512.mask.scatter.qpi.512
  return _mm512_mask_i64scatter_epi32(__addr, __mask, __index, __v1, 2); 
}

__m512d test_mm512_insertf64x4(__m512d __A, __m256d __B) {
  // CIR-LABEL: test_mm512_insertf64x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<!cir.double x 8>

  // LLVM-LABEL: test_mm512_insertf64x4
  // LLVM: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm512_insertf64x4(__A, __B, 1);
}

__m512 test_mm512_insertf32x4(__m512 __A, __m128 __B) {
  // CIR-LABEL: test_mm512_insertf32x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 16>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<!cir.float x 16>

  // LLVM-LABEL: test_mm512_insertf32x4
  // LLVM: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_insertf32x4(__A, __B, 1);
}

__m512i test_mm512_inserti64x4(__m512i __A, __m256i __B) {
  // CIR-LABEL: test_mm512_inserti64x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s64i x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_inserti64x4
  // LLVM: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm512_inserti64x4(__A, __B, 1); 
}

__m512i test_mm512_inserti32x4(__m512i __A, __m128i __B) {
  // CIR-LABEL: test_mm512_inserti32x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s32i x 16>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_inserti32x4
  // LLVM: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_inserti32x4(__A, __B, 1); 
}

__m512d test_mm512_shuffle_pd(__m512d __M, __m512d __V) {
  // CIR-LABEL: test_mm512_shuffle_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 8>) [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<3> : !s32i, #cir.int<10> : !s32i, #cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<6> : !s32i, #cir.int<14> : !s32i] : !cir.vector<!cir.double x 8>

  // CHECK-LABEL: test_mm512_shuffle_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 3, i32 10, i32 4, i32 12, i32 6, i32 14>

  // OGCG-LABEL: test_mm512_shuffle_pd
  // OGCG: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 3, i32 10, i32 4, i32 12, i32 6, i32 14>
  return _mm512_shuffle_pd(__M, __V, 4); 
}

__m512 test_mm512_shuffle_ps(__m512 __M, __m512 __V) {
  // CIR-LABEL: test_mm512_shuffle_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 16>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<16> : !s32i, #cir.int<16> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<20> : !s32i, #cir.int<20> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<24> : !s32i, #cir.int<24> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<28> : !s32i, #cir.int<28> : !s32i] : !cir.vector<!cir.float x 16>

  // CHECK-LABEL: test_mm512_shuffle_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 16, i32 16, i32 4, i32 5, i32 20, i32 20, i32 8, i32 9, i32 24, i32 24, i32 12, i32 13, i32 28, i32 28>

  // OGCG-LABEL: test_mm512_shuffle_ps
  // OGCG: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 16, i32 16, i32 4, i32 5, i32 20, i32 20, i32 8, i32 9, i32 24, i32 24, i32 12, i32 13, i32 28, i32 28>
  return _mm512_shuffle_ps(__M, __V, 4);
}

