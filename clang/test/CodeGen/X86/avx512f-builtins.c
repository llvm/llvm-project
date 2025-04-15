// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m512d test_mm512_sqrt_pd(__m512d a)
{
  // CHECK-LABEL: test_mm512_sqrt_pd
  // CHECK: call {{.*}}<8 x double> @llvm.sqrt.v8f64(<8 x double> %{{.*}})
  return _mm512_sqrt_pd(a);
}

__m512d test_mm512_mask_sqrt_pd (__m512d __W, __mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_sqrt_pd 
  // CHECK: call {{.*}}<8 x double> @llvm.sqrt.v8f64(<8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_sqrt_pd (__W,__U,__A);
}

__m512d test_mm512_maskz_sqrt_pd (__mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_maskz_sqrt_pd 
  // CHECK: call {{.*}}<8 x double> @llvm.sqrt.v8f64(<8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> {{.*}}
  return _mm512_maskz_sqrt_pd (__U,__A);
}

__m512d test_mm512_mask_sqrt_round_pd(__m512d __W,__mmask8 __U,__m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_sqrt_round_pd
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.sqrt.pd.512(<8 x double> %{{.*}}, i32 11)
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_sqrt_round_pd(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_sqrt_round_pd(__mmask8 __U,__m512d __A)
{
  // CHECK-LABEL: test_mm512_maskz_sqrt_round_pd
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.sqrt.pd.512(<8 x double> %{{.*}}, i32 11)
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> {{.*}}
  return _mm512_maskz_sqrt_round_pd(__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_sqrt_round_pd(__m512d __A)
{
  // CHECK-LABEL: test_mm512_sqrt_round_pd
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.sqrt.pd.512(<8 x double> %{{.*}}, i32 11)
  return _mm512_sqrt_round_pd(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_sqrt_ps(__m512 a)
{
  // CHECK-LABEL: test_mm512_sqrt_ps
  // CHECK: call {{.*}}<16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.*}})
  return _mm512_sqrt_ps(a);
}

__m512 test_mm512_mask_sqrt_ps(__m512 __W, __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_sqrt_ps
  // CHECK: call {{.*}}<16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_sqrt_ps( __W, __U, __A);
}

__m512 test_mm512_maskz_sqrt_ps( __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_sqrt_ps
  // CHECK: call {{.*}}<16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> {{.*}}
  return _mm512_maskz_sqrt_ps(__U ,__A);
}

__m512 test_mm512_mask_sqrt_round_ps(__m512 __W,__mmask16 __U,__m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_sqrt_round_ps
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.sqrt.ps.512(<16 x float> %{{.*}}, i32 11)
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_sqrt_round_ps(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_sqrt_round_ps(__mmask16 __U,__m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_sqrt_round_ps
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.sqrt.ps.512(<16 x float> %{{.*}}, i32 11)
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> {{.*}}
  return _mm512_maskz_sqrt_round_ps(__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_sqrt_round_ps(__m512 __A)
{
  // CHECK-LABEL: test_mm512_sqrt_round_ps
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.sqrt.ps.512(<16 x float> %{{.*}}, i32 11)
  return _mm512_sqrt_round_ps(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_rsqrt14_pd(__m512d a)
{
  // CHECK-LABEL: test_mm512_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.512
  return _mm512_rsqrt14_pd(a);
}

__m512d test_mm512_mask_rsqrt14_pd (__m512d __W, __mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_rsqrt14_pd 
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.512
  return _mm512_mask_rsqrt14_pd (__W,__U,__A);
}

__m512d test_mm512_maskz_rsqrt14_pd (__mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_maskz_rsqrt14_pd 
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.512
  return _mm512_maskz_rsqrt14_pd (__U,__A);
}

__m512 test_mm512_rsqrt14_ps(__m512 a)
{
  // CHECK-LABEL: test_mm512_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.512
  return _mm512_rsqrt14_ps(a);
}

__m512 test_mm512_mask_rsqrt14_ps (__m512 __W, __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_rsqrt14_ps 
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.512
  return _mm512_mask_rsqrt14_ps (__W,__U,__A);
}

__m512 test_mm512_maskz_rsqrt14_ps (__mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_rsqrt14_ps 
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.512
  return _mm512_maskz_rsqrt14_ps (__U,__A);
}

__m512 test_mm512_add_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: test_mm512_add_ps
  // CHECK: fadd <16 x float>
  return _mm512_add_ps(a, b);
}

__m512d test_mm512_add_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: test_mm512_add_pd
  // CHECK: fadd <8 x double>
  return _mm512_add_pd(a, b);
}

__m512 test_mm512_mul_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: test_mm512_mul_ps
  // CHECK: fmul <16 x float>
  return _mm512_mul_ps(a, b);
}

__m512d test_mm512_mul_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: test_mm512_mul_pd
  // CHECK: fmul <8 x double>
  return _mm512_mul_pd(a, b);
}

void test_mm512_storeu_si512 (void *__P, __m512i __A)
{
  // CHECK-LABEL: test_mm512_storeu_si512
  // CHECK: store <8 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm512_storeu_si512 ( __P,__A);
}

void test_mm512_storeu_ps(void *p, __m512 a)
{
  // CHECK-LABEL: test_mm512_storeu_ps
  // CHECK: store <16 x float> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm512_storeu_ps(p, a);
}

void test_mm512_storeu_pd(void *p, __m512d a)
{
  // CHECK-LABEL: test_mm512_storeu_pd
  // CHECK: store <8 x double> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm512_storeu_pd(p, a);
}

void test_mm512_mask_store_ps(void *p, __m512 a, __mmask16 m)
{
  // CHECK-LABEL: test_mm512_mask_store_ps
  // CHECK: @llvm.masked.store.v16f32.p0(<16 x float> %{{.*}}, ptr %{{.*}}, i32 64, <16 x i1> %{{.*}})
  _mm512_mask_store_ps(p, m, a);
}

void test_mm512_store_si512 (void *__P, __m512i __A)
{
  // CHECK-LABEL: test_mm512_store_si512 
  // CHECK: load <8 x i64>, ptr %__A.addr.i, align 64{{$}}
  // CHECK: [[SI512_3:%.+]] = load ptr, ptr %__P.addr.i, align 8{{$}}
  // CHECK: store <8 x i64>  
  _mm512_store_si512 ( __P,__A);
}

void test_mm512_store_epi32 (void *__P, __m512i __A)
{
  // CHECK-LABEL: test_mm512_store_epi32 
  // CHECK: load <8 x i64>, ptr %__A.addr.i, align 64{{$}}
  // CHECK: [[Si32_3:%.+]] = load ptr, ptr %__P.addr.i, align 8{{$}}
  // CHECK: store <8 x i64>  
  _mm512_store_epi32 ( __P,__A);
}

void test_mm512_store_epi64 (void *__P, __m512i __A)
{
  // CHECK-LABEL: test_mm512_store_epi64 
  // CHECK: load <8 x i64>, ptr %__A.addr.i, align 64{{$}}
  // CHECK: [[SI64_3:%.+]] = load ptr, ptr %__P.addr.i, align 8{{$}}
  // CHECK: store <8 x i64>  
  _mm512_store_epi64 ( __P,__A);
}

void test_mm512_store_ps(void *p, __m512 a)
{
  // CHECK-LABEL: test_mm512_store_ps
  // CHECK: store <16 x float>
  _mm512_store_ps(p, a);
}

void test_mm512_store_pd(void *p, __m512d a)
{
  // CHECK-LABEL: test_mm512_store_pd
  // CHECK: store <8 x double>
  _mm512_store_pd(p, a);
}

void test_mm512_mask_store_pd(void *p, __m512d a, __mmask8 m)
{
  // CHECK-LABEL: test_mm512_mask_store_pd
  // CHECK: @llvm.masked.store.v8f64.p0(<8 x double> %{{.*}}, ptr %{{.*}}, i32 64, <8 x i1> %{{.*}})
  _mm512_mask_store_pd(p, m, a);
}

void test_mm512_storeu_epi32(void *__P, __m512i __A) {
  // CHECK-LABEL: test_mm512_storeu_epi32
  // CHECK: store <8 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm512_storeu_epi32(__P, __A); 
}

void test_mm512_mask_storeu_epi32(void *__P, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_storeu_epi32
  // CHECK: @llvm.masked.store.v16i32.p0(<16 x i32> %{{.*}}, ptr %{{.*}}, i32 1, <16 x i1> %{{.*}})
  return _mm512_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm512_storeu_epi64(void *__P, __m512i __A) {
  // CHECK-LABEL: test_mm512_storeu_epi64
  // CHECK: store <8 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm512_storeu_epi64(__P, __A); 
}

void test_mm512_mask_storeu_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_storeu_epi64
  // CHECK: @llvm.masked.store.v8i64.p0(<8 x i64> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm512_mask_storeu_epi64(__P, __U, __A); 
}

__m512i test_mm512_loadu_si512 (void *__P)
{
  // CHECK-LABEL: test_mm512_loadu_si512 
  // CHECK: load <8 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm512_loadu_si512 ( __P);
}

__m512i test_mm512_loadu_epi32 (void *__P)
{
  // CHECK-LABEL: test_mm512_loadu_epi32 
  // CHECK: load <8 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm512_loadu_epi32 (__P);
}

__m512i test_mm512_mask_loadu_epi32 (__m512i __W, __mmask16 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_mask_loadu_epi32 
  // CHECK: @llvm.masked.load.v16i32.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_mask_loadu_epi32 (__W,__U, __P);
}

__m512i test_mm512_maskz_loadu_epi32 (__mmask16 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_maskz_loadu_epi32
  // CHECK: @llvm.masked.load.v16i32.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_maskz_loadu_epi32 (__U, __P);
}

__m512i test_mm512_loadu_epi64 (void *__P)
{
  // CHECK-LABEL: test_mm512_loadu_epi64 
  // CHECK: load <8 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm512_loadu_epi64 (__P);
}

__m512i test_mm512_mask_loadu_epi64 (__m512i __W, __mmask8 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_mask_loadu_epi64 
  // CHECK: @llvm.masked.load.v8i64.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_mask_loadu_epi64 (__W,__U, __P);
}

__m512i test_mm512_maskz_loadu_epi64 (__mmask16 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_maskz_loadu_epi64
  // CHECK: @llvm.masked.load.v8i64.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_maskz_loadu_epi64 (__U, __P);
}

__m512 test_mm512_loadu_ps(void *p)
{
  // CHECK-LABEL: test_mm512_loadu_ps
  // CHECK: load <16 x float>, ptr {{.*}}, align 1{{$}}
  return _mm512_loadu_ps(p);
}

__m512 test_mm512_mask_loadu_ps (__m512 __W, __mmask16 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_mask_loadu_ps 
  // CHECK: @llvm.masked.load.v16f32.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_mask_loadu_ps (__W,__U, __P);
}

__m512d test_mm512_loadu_pd(void *p)
{
  // CHECK-LABEL: test_mm512_loadu_pd
  // CHECK: load <8 x double>, ptr {{.*}}, align 1{{$}}
  return _mm512_loadu_pd(p);
}

__m512d test_mm512_mask_loadu_pd (__m512d __W, __mmask8 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_mask_loadu_pd 
  // CHECK: @llvm.masked.load.v8f64.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_mask_loadu_pd (__W,__U, __P);
}

__m512i test_mm512_load_si512 (void *__P)
{
  // CHECK-LABEL: test_mm512_load_si512 
  // CHECK: [[LI512_1:%.+]] = load ptr, ptr %__P.addr.i, align 8{{$}}
  // CHECK: load <8 x i64>, ptr [[LI512_1]], align 64{{$}}
  return _mm512_load_si512 ( __P);
}

__m512i test_mm512_load_epi32 (void *__P)
{
  // CHECK-LABEL: test_mm512_load_epi32 
  // CHECK: [[LI32_1:%.+]] = load ptr, ptr %__P.addr.i, align 8{{$}}
  // CHECK: load <8 x i64>, ptr [[LI32_1]], align 64{{$}}
  return _mm512_load_epi32 ( __P);
}

__m512i test_mm512_load_epi64 (void *__P)
{
  // CHECK-LABEL: test_mm512_load_epi64 
  // CHECK: [[LI64_1:%.+]] = load ptr, ptr %__P.addr.i, align 8{{$}}
  // CHECK: load <8 x i64>, ptr [[LI64_1]], align 64{{$}}
  return _mm512_load_epi64 ( __P);
}

__m512 test_mm512_load_ps(void *p)
{
  // CHECK-LABEL: test_mm512_load_ps
  // CHECK: load <16 x float>, ptr %{{.*}}, align 64{{$}}
  return _mm512_load_ps(p);
}

__m512 test_mm512_mask_load_ps (__m512 __W, __mmask16 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_mask_load_ps 
  // CHECK: @llvm.masked.load.v16f32.p0(ptr %{{.*}}, i32 64, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_mask_load_ps (__W,__U, __P);
}

__m512 test_mm512_maskz_load_ps(__mmask16 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_maskz_load_ps
  // CHECK: @llvm.masked.load.v16f32.p0(ptr %{{.*}}, i32 64, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_maskz_load_ps(__U, __P);
}

__m512d test_mm512_load_pd(void *p)
{
  // CHECK-LABEL: test_mm512_load_pd
  // CHECK: load <8 x double>, ptr %{{.*}}, align 64{{$}}
  return _mm512_load_pd(p);
}

__m512d test_mm512_mask_load_pd (__m512d __W, __mmask8 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_mask_load_pd 
  // CHECK: @llvm.masked.load.v8f64.p0(ptr %{{.*}}, i32 64, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_mask_load_pd (__W,__U, __P);
}

__m512d test_mm512_maskz_load_pd(__mmask8 __U, void *__P)
{
  // CHECK-LABEL: test_mm512_maskz_load_pd
  // CHECK: @llvm.masked.load.v8f64.p0(ptr %{{.*}}, i32 64, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_maskz_load_pd(__U, __P);
}

__m512d test_mm512_set1_pd(double d)
{
  // CHECK-LABEL: test_mm512_set1_pd
  // CHECK: insertelement <8 x double> {{.*}}, i32 0
  // CHECK: insertelement <8 x double> {{.*}}, i32 1
  // CHECK: insertelement <8 x double> {{.*}}, i32 2
  // CHECK: insertelement <8 x double> {{.*}}, i32 3
  // CHECK: insertelement <8 x double> {{.*}}, i32 4
  // CHECK: insertelement <8 x double> {{.*}}, i32 5
  // CHECK: insertelement <8 x double> {{.*}}, i32 6
  // CHECK: insertelement <8 x double> {{.*}}, i32 7
  return _mm512_set1_pd(d);
}

__mmask16 test_mm512_knot(__mmask16 a)
{
  // CHECK-LABEL: test_mm512_knot
  // CHECK: [[IN:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[NOT:%.*]] = xor <16 x i1> [[IN]], splat (i1 true)
  // CHECK: bitcast <16 x i1> [[NOT]] to i16
  return _mm512_knot(a);
}

__m512i test_mm512_alignr_epi32(__m512i a, __m512i b)
{
  // CHECK-LABEL: test_mm512_alignr_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  return _mm512_alignr_epi32(a, b, 2);
}

__m512i test_mm512_mask_alignr_epi32(__m512i w, __mmask16 u, __m512i a, __m512i b)
{
  // CHECK-LABEL: test_mm512_mask_alignr_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> {{.*}}
  return _mm512_mask_alignr_epi32(w, u, a, b, 2);
}

__m512i test_mm512_maskz_alignr_epi32( __mmask16 u, __m512i a, __m512i b)
{
  // CHECK-LABEL: test_mm512_maskz_alignr_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> {{.*}}
  return _mm512_maskz_alignr_epi32(u, a, b, 2);
}

__m512i test_mm512_alignr_epi64(__m512i a, __m512i b)
{
  // CHECK-LABEL: test_mm512_alignr_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  return _mm512_alignr_epi64(a, b, 2);
}

__m512i test_mm512_mask_alignr_epi64(__m512i w, __mmask8 u, __m512i a, __m512i b)
{
  // CHECK-LABEL: test_mm512_mask_alignr_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> {{.*}}
  return _mm512_mask_alignr_epi64(w, u, a, b, 2);
}

__m512i test_mm512_maskz_alignr_epi64( __mmask8 u, __m512i a, __m512i b)
{
  // CHECK-LABEL: test_mm512_maskz_alignr_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> {{.*}}
  return _mm512_maskz_alignr_epi64(u, a, b, 2);
}

__m512d test_mm512_fmadd_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fmadd_round_pd
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  return _mm512_fmadd_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_fmadd_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fmadd_round_pd
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fmadd_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask3_fmadd_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmadd_round_pd
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fmadd_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_fmadd_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fmadd_round_pd
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fmadd_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_fmsub_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fmsub_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  return _mm512_fmsub_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_fmsub_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fmsub_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fmsub_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_fmsub_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsub_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fmsub_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_fnmadd_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fnmadd_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  return _mm512_fnmadd_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask3_fnmadd_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmadd_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fnmadd_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_fnmadd_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmadd_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fnmadd_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_fnmsub_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fnmsub_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  return _mm512_fnmsub_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_fnmsub_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmsub_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fnmsub_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_fmadd_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fmadd_pd
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_fmadd_pd(__A, __B, __C);
}
__m512d test_mm512_mask_fmadd_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fmadd_pd
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fmadd_pd(__A, __U, __B, __C);
}
__m512d test_mm512_mask3_fmadd_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmadd_pd
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fmadd_pd(__A, __B, __C, __U);
}
__m512d test_mm512_maskz_fmadd_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fmadd_pd
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fmadd_pd(__U, __A, __B, __C);
}
__m512d test_mm512_fmsub_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fmsub_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_fmsub_pd(__A, __B, __C);
}
__m512d test_mm512_mask_fmsub_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fmsub_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fmsub_pd(__A, __U, __B, __C);
}
__m512d test_mm512_maskz_fmsub_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsub_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fmsub_pd(__U, __A, __B, __C);
}
__m512d test_mm512_fnmadd_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fnmadd_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_fnmadd_pd(__A, __B, __C);
}
__m512d test_mm512_mask3_fnmadd_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmadd_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fnmadd_pd(__A, __B, __C, __U);
}
__m512d test_mm512_maskz_fnmadd_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmadd_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fnmadd_pd(__U, __A, __B, __C);
}
__m512d test_mm512_fnmsub_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fnmsub_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_fnmsub_pd(__A, __B, __C);
}
__m512d test_mm512_maskz_fnmsub_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmsub_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fnmsub_pd(__U, __A, __B, __C);
}
__m512 test_mm512_fmadd_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fmadd_round_ps
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  return _mm512_fmadd_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_fmadd_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fmadd_round_ps
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fmadd_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask3_fmadd_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmadd_round_ps
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmadd_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_fmadd_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fmadd_round_ps
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fmadd_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_fmsub_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fmsub_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  return _mm512_fmsub_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_fmsub_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fmsub_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fmsub_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_fmsub_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsub_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fmsub_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_fnmadd_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fnmadd_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  return _mm512_fnmadd_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask3_fnmadd_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmadd_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fnmadd_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_fnmadd_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmadd_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fnmadd_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_fnmsub_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fnmsub_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  return _mm512_fnmsub_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_fnmsub_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmsub_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fnmsub_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_fmadd_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fmadd_ps
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_fmadd_ps(__A, __B, __C);
}
__m512 test_mm512_mask_fmadd_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fmadd_ps
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_mask_fmadd_ps(__A, __U, __B, __C);
}
__m512 test_mm512_mask3_fmadd_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmadd_ps
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmadd_ps(__A, __B, __C, __U);
}
__m512 test_mm512_maskz_fmadd_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fmadd_ps
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fmadd_ps(__U, __A, __B, __C);
}
__m512 test_mm512_fmsub_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fmsub_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_fmsub_ps(__A, __B, __C);
}
__m512 test_mm512_mask_fmsub_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fmsub_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fmsub_ps(__A, __U, __B, __C);
}
__m512 test_mm512_maskz_fmsub_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsub_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fmsub_ps(__U, __A, __B, __C);
}
__m512 test_mm512_fnmadd_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fnmadd_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_fnmadd_ps(__A, __B, __C);
}
__m512 test_mm512_mask3_fnmadd_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmadd_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fnmadd_ps(__A, __B, __C, __U);
}
__m512 test_mm512_maskz_fnmadd_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmadd_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fnmadd_ps(__U, __A, __B, __C);
}
__m512 test_mm512_fnmsub_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fnmsub_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_fnmsub_ps(__A, __B, __C);
}
__m512 test_mm512_maskz_fnmsub_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmsub_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fnmsub_ps(__U, __A, __B, __C);
}
__m512d test_mm512_fmaddsub_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fmaddsub_round_pd
  // CHECK: @llvm.x86.avx512.vfmaddsub.pd.512
  return _mm512_fmaddsub_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_fmaddsub_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fmaddsub_round_pd
  // CHECK: @llvm.x86.avx512.vfmaddsub.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fmaddsub_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask3_fmaddsub_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmaddsub_round_pd
  // CHECK: @llvm.x86.avx512.vfmaddsub.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fmaddsub_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_fmaddsub_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fmaddsub_round_pd
  // CHECK: @llvm.x86.avx512.vfmaddsub.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fmaddsub_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_fmsubadd_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fmsubadd_round_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmaddsub.pd.512
  return _mm512_fmsubadd_round_pd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_fmsubadd_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fmsubadd_round_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmaddsub.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fmsubadd_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_fmsubadd_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsubadd_round_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmaddsub.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fmsubadd_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_fmaddsub_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.vfmaddsub.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4)
  return _mm512_fmaddsub_pd(__A, __B, __C);
}
__m512d test_mm512_mask_fmaddsub_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.vfmaddsub.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4)
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fmaddsub_pd(__A, __U, __B, __C);
}
__m512d test_mm512_mask3_fmaddsub_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.vfmaddsub.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4)
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fmaddsub_pd(__A, __B, __C, __U);
}
__m512d test_mm512_maskz_fmaddsub_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.vfmaddsub.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4)
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fmaddsub_pd(__U, __A, __B, __C);
}
__m512d test_mm512_fmsubadd_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.vfmaddsub.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> [[NEG]], i32 4)
  return _mm512_fmsubadd_pd(__A, __B, __C);
}
__m512d test_mm512_mask_fmsubadd_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.vfmaddsub.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> [[NEG]], i32 4)
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fmsubadd_pd(__A, __U, __B, __C);
}
__m512d test_mm512_maskz_fmsubadd_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.vfmaddsub.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> [[NEG]], i32 4)
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> zeroinitializer
  return _mm512_maskz_fmsubadd_pd(__U, __A, __B, __C);
}
__m512 test_mm512_fmaddsub_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fmaddsub_round_ps
  // CHECK: @llvm.x86.avx512.vfmaddsub.ps.512
  return _mm512_fmaddsub_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_fmaddsub_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fmaddsub_round_ps
  // CHECK: @llvm.x86.avx512.vfmaddsub.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fmaddsub_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask3_fmaddsub_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmaddsub_round_ps
  // CHECK: @llvm.x86.avx512.vfmaddsub.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmaddsub_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_fmaddsub_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fmaddsub_round_ps
  // CHECK: @llvm.x86.avx512.vfmaddsub.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fmaddsub_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_fmsubadd_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fmsubadd_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmaddsub.ps.512
  return _mm512_fmsubadd_round_ps(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_fmsubadd_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fmsubadd_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmaddsub.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fmsubadd_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_fmsubadd_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsubadd_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmaddsub.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fmsubadd_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_fmaddsub_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.vfmaddsub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  return _mm512_fmaddsub_ps(__A, __B, __C);
}
__m512 test_mm512_mask_fmaddsub_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.vfmaddsub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fmaddsub_ps(__A, __U, __B, __C);
}
__m512 test_mm512_mask3_fmaddsub_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.vfmaddsub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmaddsub_ps(__A, __B, __C, __U);
}
__m512 test_mm512_maskz_fmaddsub_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.vfmaddsub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fmaddsub_ps(__U, __A, __B, __C);
}
__m512 test_mm512_fmsubadd_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.vfmaddsub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> [[NEG]], i32 4)
  return _mm512_fmsubadd_ps(__A, __B, __C);
}
__m512 test_mm512_mask_fmsubadd_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.vfmaddsub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> [[NEG]], i32 4)
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fmsubadd_ps(__A, __U, __B, __C);
}
__m512 test_mm512_maskz_fmsubadd_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.vfmaddsub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> [[NEG]], i32 4)
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> zeroinitializer
  return _mm512_maskz_fmsubadd_ps(__U, __A, __B, __C);
}
__m512d test_mm512_mask3_fmsub_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsub_round_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fmsub_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask3_fmsub_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsub_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fmsub_pd(__A, __B, __C, __U);
}
__m512 test_mm512_mask3_fmsub_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsub_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmsub_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask3_fmsub_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsub_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmsub_ps(__A, __B, __C, __U);
}
__m512d test_mm512_mask3_fmsubadd_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsubadd_round_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmaddsub.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fmsubadd_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask3_fmsubadd_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.x86.avx512.vfmaddsub.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> [[NEG]], i32 4)
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fmsubadd_pd(__A, __B, __C, __U);
}
__m512 test_mm512_mask3_fmsubadd_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsubadd_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmaddsub.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmsubadd_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask3_fmsubadd_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.x86.avx512.vfmaddsub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> [[NEG]], i32 4)
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmsubadd_ps(__A, __B, __C, __U);
}
__m512d test_mm512_mask_fnmadd_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fnmadd_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fnmadd_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_fnmadd_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fnmadd_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fnmadd_pd(__A, __U, __B, __C);
}
__m512 test_mm512_mask_fnmadd_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fnmadd_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fnmadd_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_fnmadd_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fnmadd_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fnmadd_ps(__A, __U, __B, __C);
}
__m512d test_mm512_mask_fnmsub_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fnmsub_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fnmsub_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask3_fnmsub_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmsub_round_pd
  // CHECK: fneg <8 x double>
  // CHECK: fneg <8 x double>
  // CHECK: @llvm.x86.avx512.vfmadd.pd.512
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fnmsub_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_fnmsub_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: test_mm512_mask_fnmsub_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_fnmsub_pd(__A, __U, __B, __C);
}
__m512d test_mm512_mask3_fnmsub_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmsub_pd
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: fneg <8 x double> %{{.*}}
  // CHECK: call {{.*}}<8 x double> @llvm.fma.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask3_fnmsub_pd(__A, __B, __C, __U);
}
__m512 test_mm512_mask_fnmsub_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fnmsub_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fnmsub_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask3_fnmsub_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmsub_round_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: @llvm.x86.avx512.vfmadd.ps.512
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fnmsub_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_fnmsub_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: test_mm512_mask_fnmsub_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fnmsub_ps(__A, __U, __B, __C);
}
__m512 test_mm512_mask3_fnmsub_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmsub_ps
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: fneg <16 x float> %{{.*}}
  // CHECK: call {{.*}}<16 x float> @llvm.fma.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fnmsub_ps(__A, __B, __C, __U);
}

__mmask16 test_mm512_cmpeq_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpeq_epi32_mask
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmpeq_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpeq_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpeq_epi32_mask
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_mask_cmpeq_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpeq_epi64_mask
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpeq_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpeq_epi64_mask
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmpeq_epi64_mask(__a, __b);
}

__mmask16 test_mm512_cmpgt_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpgt_epi32_mask
  // CHECK: icmp sgt <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmpgt_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpgt_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpgt_epi32_mask
  // CHECK: icmp sgt <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_mask_cmpgt_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpgt_epi64_mask
  // CHECK: icmp sgt <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpgt_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpgt_epi64_mask
  // CHECK: icmp sgt <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmpgt_epi64_mask(__a, __b);
}

__m512d test_mm512_unpackhi_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: test_mm512_unpackhi_pd
  // CHECK: shufflevector <8 x double> {{.*}} <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  return _mm512_unpackhi_pd(a, b);
}

__m512d test_mm512_unpacklo_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: test_mm512_unpacklo_pd
  // CHECK: shufflevector <8 x double> {{.*}} <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  return _mm512_unpacklo_pd(a, b);
}

__m512 test_mm512_unpackhi_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: test_mm512_unpackhi_ps
  // CHECK: shufflevector <16 x float> {{.*}} <i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  return _mm512_unpackhi_ps(a, b);
}

__m512 test_mm512_unpacklo_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: test_mm512_unpacklo_ps
  // CHECK: shufflevector <16 x float> {{.*}} <i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  return _mm512_unpacklo_ps(a, b);
}

__mmask16 test_mm512_cmp_round_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_round_ps_mask
  // CHECK: fcmp oeq <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_round_ps_mask(a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask16 test_mm512_mask_cmp_round_ps_mask(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_round_ps_mask
  // CHECK: [[CMP:%.*]] = fcmp oeq <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_round_ps_mask(m, a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask16 test_mm512_cmp_ps_mask_eq_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_eq_oq
  // CHECK: fcmp oeq <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_lt_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_lt_os
  // CHECK: fcmp olt <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask16 test_mm512_cmp_ps_mask_le_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_le_os
  // CHECK: fcmp ole <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_LE_OS);
}

__mmask16 test_mm512_cmp_ps_mask_unord_q(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_unord_q
  // CHECK: fcmp uno <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
}

__mmask16 test_mm512_cmp_ps_mask_neq_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_neq_uq
  // CHECK: fcmp une <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_nlt_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nlt_us
  // CHECK: fcmp uge <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NLT_US);
}

__mmask16 test_mm512_cmp_ps_mask_nle_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nle_us
  // CHECK: fcmp ugt <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NLE_US);
}

__mmask16 test_mm512_cmp_ps_mask_ord_q(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ord_q
  // CHECK: fcmp ord <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_ORD_Q);
}

__mmask16 test_mm512_cmp_ps_mask_eq_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_eq_uq
  // CHECK: fcmp ueq <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_nge_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nge_us
  // CHECK: fcmp ult <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NGE_US);
}

__mmask16 test_mm512_cmp_ps_mask_ngt_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ngt_us
  // CHECK: fcmp ule <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NGT_US);
}

__mmask16 test_mm512_cmp_ps_mask_false_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_false_oq
  // CHECK: fcmp false <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_FALSE_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_neq_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_neq_oq
  // CHECK: fcmp one <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_ge_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ge_os
  // CHECK: fcmp oge <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_GE_OS);
}

__mmask16 test_mm512_cmp_ps_mask_gt_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_gt_os
  // CHECK: fcmp ogt <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_GT_OS);
}

__mmask16 test_mm512_cmp_ps_mask_true_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_true_uq
  // CHECK: fcmp true <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_TRUE_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_eq_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_eq_os
  // CHECK: fcmp oeq <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OS);
}

__mmask16 test_mm512_cmp_ps_mask_lt_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_lt_oq
  // CHECK: fcmp olt <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_le_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_le_oq
  // CHECK: fcmp ole <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_unord_s(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_unord_s
  // CHECK: fcmp uno <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_UNORD_S);
}

__mmask16 test_mm512_cmp_ps_mask_neq_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_neq_us
  // CHECK: fcmp une <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_US);
}

__mmask16 test_mm512_cmp_ps_mask_nlt_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nlt_uq
  // CHECK: fcmp uge <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NLT_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_nle_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nle_uq
  // CHECK: fcmp ugt <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NLE_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_ord_s(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ord_s
  // CHECK: fcmp ord <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_ORD_S);
}

__mmask16 test_mm512_cmp_ps_mask_eq_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_eq_us
  // CHECK: fcmp ueq <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_US);
}

__mmask16 test_mm512_cmp_ps_mask_nge_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_nge_uq
  // CHECK: fcmp ult <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NGE_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_ngt_uq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ngt_uq
  // CHECK: fcmp ule <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NGT_UQ);
}

__mmask16 test_mm512_cmp_ps_mask_false_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_false_os
  // CHECK: fcmp false <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_FALSE_OS);
}

__mmask16 test_mm512_cmp_ps_mask_neq_os(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_neq_os
  // CHECK: fcmp one <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OS);
}

__mmask16 test_mm512_cmp_ps_mask_ge_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_ge_oq
  // CHECK: fcmp oge <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_gt_oq(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_gt_oq
  // CHECK: fcmp ogt <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
}

__mmask16 test_mm512_cmp_ps_mask_true_us(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmp_ps_mask_true_us
  // CHECK: fcmp true <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmp_ps_mask(a, b, _CMP_TRUE_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_eq_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_eq_oq
  // CHECK: [[CMP:%.*]] = fcmp oeq <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_lt_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_lt_os
  // CHECK: [[CMP:%.*]] = fcmp olt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_le_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_le_os
  // CHECK: [[CMP:%.*]] = fcmp ole <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LE_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_unord_q(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_unord_q
  // CHECK: [[CMP:%.*]] = fcmp uno <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask16 test_mm512_mask_cmp_ps_mask_neq_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_neq_uq
  // CHECK: [[CMP:%.*]] = fcmp une <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nlt_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nlt_us
  // CHECK: [[CMP:%.*]] = fcmp uge <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nle_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nle_us
  // CHECK: [[CMP:%.*]] = fcmp ugt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLE_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ord_q(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ord_q
  // CHECK: [[CMP:%.*]] = fcmp ord <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_ORD_Q);
}

__mmask16 test_mm512_mask_cmp_ps_mask_eq_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_eq_uq
  // CHECK: [[CMP:%.*]] = fcmp ueq <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nge_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nge_us
  // CHECK: [[CMP:%.*]] = fcmp ult <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NGE_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ngt_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ngt_us
  // CHECK: [[CMP:%.*]] = fcmp ule <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NGT_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_false_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_false_oq
  // CHECK: [[CMP:%.*]] = fcmp false <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_neq_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_neq_oq
  // CHECK: [[CMP:%.*]] = fcmp one <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ge_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ge_os
  // CHECK: [[CMP:%.*]] = fcmp oge <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_GE_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_gt_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_gt_os
  // CHECK: [[CMP:%.*]] = fcmp ogt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_GT_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_true_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_true_uq
  // CHECK: [[CMP:%.*]] = fcmp true <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_eq_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_eq_os
  // CHECK: [[CMP:%.*]] = fcmp oeq <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_lt_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_lt_oq
  // CHECK: [[CMP:%.*]] = fcmp olt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LT_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_le_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_le_oq
  // CHECK: [[CMP:%.*]] = fcmp ole <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_LE_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_unord_s(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_unord_s
  // CHECK: [[CMP:%.*]] = fcmp uno <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_S);
}

__mmask16 test_mm512_mask_cmp_ps_mask_neq_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_neq_us
  // CHECK: [[CMP:%.*]] = fcmp une <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nlt_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = fcmp uge <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nle_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nle_uq
  // CHECK: [[CMP:%.*]] = fcmp ugt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ord_s(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ord_s
  // CHECK: [[CMP:%.*]] = fcmp ord <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_ORD_S);
}

__mmask16 test_mm512_mask_cmp_ps_mask_eq_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_eq_us
  // CHECK: [[CMP:%.*]] = fcmp ueq <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_EQ_US);
}

__mmask16 test_mm512_mask_cmp_ps_mask_nge_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_nge_uq
  // CHECK: [[CMP:%.*]] = fcmp ult <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ngt_uq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = fcmp ule <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_false_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_false_os
  // CHECK: [[CMP:%.*]] = fcmp false <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_neq_os(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_neq_os
  // CHECK: [[CMP:%.*]] = fcmp one <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask16 test_mm512_mask_cmp_ps_mask_ge_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_ge_oq
  // CHECK: [[CMP:%.*]] = fcmp oge <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_GE_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_gt_oq(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_gt_oq
  // CHECK: [[CMP:%.*]] = fcmp ogt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_GT_OQ);
}

__mmask16 test_mm512_mask_cmp_ps_mask_true_us(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ps_mask_true_us
  // CHECK: [[CMP:%.*]] = fcmp true <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm512_cmp_round_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_round_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_round_pd_mask(a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm512_mask_cmp_round_pd_mask(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_round_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_round_pd_mask(m, a, b, _CMP_EQ_OQ, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm512_cmp_pd_mask_eq_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_eq_oq
  // CHECK: fcmp oeq <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_lt_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_lt_os
  // CHECK: fcmp olt <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm512_cmp_pd_mask_le_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_le_os
  // CHECK: fcmp ole <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm512_cmp_pd_mask_unord_q(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_unord_q
  // CHECK: fcmp uno <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm512_cmp_pd_mask_neq_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_neq_uq
  // CHECK: fcmp une <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_nlt_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nlt_us
  // CHECK: fcmp uge <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm512_cmp_pd_mask_nle_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nle_us
  // CHECK: fcmp ugt <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm512_cmp_pd_mask_ord_q(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ord_q
  // CHECK: fcmp ord <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm512_cmp_pd_mask_eq_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_eq_uq
  // CHECK: fcmp ueq <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_nge_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nge_us
  // CHECK: fcmp ult <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm512_cmp_pd_mask_ngt_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ngt_us
  // CHECK: fcmp ule <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm512_cmp_pd_mask_false_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_false_oq
  // CHECK: fcmp false <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_neq_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_neq_oq
  // CHECK: fcmp one <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_ge_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ge_os
  // CHECK: fcmp oge <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm512_cmp_pd_mask_gt_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_gt_os
  // CHECK: fcmp ogt <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm512_cmp_pd_mask_true_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_true_uq
  // CHECK: fcmp true <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_eq_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_eq_os
  // CHECK: fcmp oeq <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm512_cmp_pd_mask_lt_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_lt_oq
  // CHECK: fcmp olt <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_le_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_le_oq
  // CHECK: fcmp ole <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_unord_s(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_unord_s
  // CHECK: fcmp uno <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm512_cmp_pd_mask_neq_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_neq_us
  // CHECK: fcmp une <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm512_cmp_pd_mask_nlt_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nlt_uq
  // CHECK: fcmp uge <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_nle_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nle_uq
  // CHECK: fcmp ugt <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_ord_s(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ord_s
  // CHECK: fcmp ord <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm512_cmp_pd_mask_eq_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_eq_us
  // CHECK: fcmp ueq <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm512_cmp_pd_mask_nge_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_nge_uq
  // CHECK: fcmp ult <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_ngt_uq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ngt_uq
  // CHECK: fcmp ule <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm512_cmp_pd_mask_false_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_false_os
  // CHECK: fcmp false <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm512_cmp_pd_mask_neq_os(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_neq_os
  // CHECK: fcmp one <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm512_cmp_pd_mask_ge_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_ge_oq
  // CHECK: fcmp oge <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_gt_oq(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_gt_oq
  // CHECK: fcmp ogt <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm512_cmp_pd_mask_true_us(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmp_pd_mask_true_us
  // CHECK: fcmp true <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmp_pd_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_eq_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_eq_oq
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_lt_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_lt_os
  // CHECK: [[CMP:%.*]] = fcmp olt <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_le_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_le_os
  // CHECK: [[CMP:%.*]] = fcmp ole <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_unord_q(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_unord_q
  // CHECK: [[CMP:%.*]] = fcmp uno <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm512_mask_cmp_pd_mask_neq_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_neq_uq
  // CHECK: [[CMP:%.*]] = fcmp une <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nlt_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nlt_us
  // CHECK: [[CMP:%.*]] = fcmp uge <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nle_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nle_us
  // CHECK: [[CMP:%.*]] = fcmp ugt <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ord_q(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ord_q
  // CHECK: [[CMP:%.*]] = fcmp ord <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm512_mask_cmp_pd_mask_eq_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_eq_uq
  // CHECK: [[CMP:%.*]] = fcmp ueq <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nge_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nge_us
  // CHECK: [[CMP:%.*]] = fcmp ult <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ngt_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ngt_us
  // CHECK: [[CMP:%.*]] = fcmp ule <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_false_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_false_oq
  // CHECK: [[CMP:%.*]] = fcmp false <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_neq_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_neq_oq
  // CHECK: [[CMP:%.*]] = fcmp one <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ge_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ge_os
  // CHECK: [[CMP:%.*]] = fcmp oge <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_gt_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_gt_os
  // CHECK: [[CMP:%.*]] = fcmp ogt <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_true_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_true_uq
  // CHECK: [[CMP:%.*]] = fcmp true <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_eq_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_eq_os
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_lt_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_lt_oq
  // CHECK: [[CMP:%.*]] = fcmp olt <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_le_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_le_oq
  // CHECK: [[CMP:%.*]] = fcmp ole <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_unord_s(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_unord_s
  // CHECK: [[CMP:%.*]] = fcmp uno <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm512_mask_cmp_pd_mask_neq_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_neq_us
  // CHECK: [[CMP:%.*]] = fcmp une <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nlt_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = fcmp uge <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nle_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nle_uq
  // CHECK: [[CMP:%.*]] = fcmp ugt <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ord_s(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ord_s
  // CHECK: [[CMP:%.*]] = fcmp ord <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm512_mask_cmp_pd_mask_eq_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_eq_us
  // CHECK: [[CMP:%.*]] = fcmp ueq <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask_nge_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_nge_uq
  // CHECK: [[CMP:%.*]] = fcmp ult <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ngt_uq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = fcmp ule <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_false_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_false_os
  // CHECK: [[CMP:%.*]] = fcmp false <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_neq_os(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_neq_os
  // CHECK: [[CMP:%.*]] = fcmp one <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm512_mask_cmp_pd_mask_ge_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_ge_oq
  // CHECK: [[CMP:%.*]] = fcmp oge <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_gt_oq(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_gt_oq
  // CHECK: [[CMP:%.*]] = fcmp ogt <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm512_mask_cmp_pd_mask_true_us(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask_true_us
  // CHECK: [[CMP:%.*]] = fcmp true <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm512_mask_cmp_pd_mask(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_pd_mask(m, a, b, 0);
}

__mmask8 test_mm512_cmpeq_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmpeq_pd_mask
  // CHECK: fcmp oeq <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmpeq_pd_mask(a, b);
}

__mmask16 test_mm512_cmpeq_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmpeq_ps_mask
  // CHECK: fcmp oeq <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmpeq_ps_mask(a, b);
}

__mmask8 test_mm512_mask_cmpeq_pd_mask(__mmask8 k, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmpeq_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpeq_pd_mask(k, a, b);
}

__mmask16 test_mm512_mask_cmpeq_ps_mask(__mmask16 k, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmpeq_ps_mask
  // CHECK: [[CMP:%.*]] = fcmp oeq <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpeq_ps_mask(k, a, b);
}

__mmask8 test_mm512_cmple_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmple_pd_mask
  // CHECK: fcmp ole <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmple_pd_mask(a, b);
}

__mmask16 test_mm512_cmple_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmple_ps_mask
  // CHECK: fcmp ole <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmple_ps_mask(a, b);
}

__mmask8 test_mm512_mask_cmple_pd_mask(__mmask8 k, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmple_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp ole <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmple_pd_mask(k, a, b);
}

__mmask16 test_mm512_mask_cmple_ps_mask(__mmask16 k, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmple_ps_mask
  // CHECK: [[CMP:%.*]] = fcmp ole <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmple_ps_mask(k, a, b);
}

__mmask8 test_mm512_cmplt_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmplt_pd_mask
  // CHECK: fcmp olt <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmplt_pd_mask(a, b);
}

__mmask16 test_mm512_cmplt_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmplt_ps_mask
  // CHECK: fcmp olt <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmplt_ps_mask(a, b);
}

__mmask8 test_mm512_mask_cmplt_pd_mask(__mmask8 k, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmplt_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp olt <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmplt_pd_mask(k, a, b);
}

__mmask16 test_mm512_mask_cmplt_ps_mask(__mmask16 k, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmplt_ps_mask
  // CHECK: [[CMP:%.*]] = fcmp olt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmplt_ps_mask(k, a, b);
}

__mmask8 test_mm512_cmpneq_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmpneq_pd_mask
  // CHECK: fcmp une <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmpneq_pd_mask(a, b);
}

__mmask16 test_mm512_cmpneq_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmpneq_ps_mask
  // CHECK: fcmp une <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmpneq_ps_mask(a, b);
}

__mmask8 test_mm512_mask_cmpneq_pd_mask(__mmask8 k, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmpneq_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp une <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpneq_pd_mask(k, a, b);
}

__mmask16 test_mm512_mask_cmpneq_ps_mask(__mmask16 k, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmpneq_ps_mask
  // CHECK: [[CMP:%.*]] = fcmp une <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpneq_ps_mask(k, a, b);
}

__mmask8 test_mm512_cmpnle_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmpnle_pd_mask
  // CHECK: fcmp ugt <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmpnle_pd_mask(a, b);
}

__mmask16 test_mm512_cmpnle_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmpnle_ps_mask
  // CHECK: fcmp ugt <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmpnle_ps_mask(a, b);
}

__mmask8 test_mm512_mask_cmpnle_pd_mask(__mmask8 k, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmpnle_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp ugt <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpnle_pd_mask(k, a, b);
}

__mmask16 test_mm512_mask_cmpnle_ps_mask(__mmask16 k, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmpnle_ps_mask
  // CHECK: [[CMP:%.*]] = fcmp ugt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpnle_ps_mask(k, a, b);
}

__mmask8 test_mm512_cmpnlt_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmpnlt_pd_mask
  // CHECK: fcmp uge <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmpnlt_pd_mask(a, b);
}

__mmask16 test_mm512_cmpnlt_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmpnlt_ps_mask
  // CHECK: fcmp uge <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmpnlt_ps_mask(a, b);
}

__mmask8 test_mm512_mask_cmpnlt_pd_mask(__mmask8 k, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmpnlt_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp uge <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpnlt_pd_mask(k, a, b);
}

__mmask16 test_mm512_mask_cmpnlt_ps_mask(__mmask16 k, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmpnlt_ps_mask
  // CHECK: [[CMP:%.*]] = fcmp uge <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpnlt_ps_mask(k, a, b);
}

__mmask8 test_mm512_cmpord_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmpord_pd_mask
  // CHECK: fcmp ord <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmpord_pd_mask(a, b);
}

__mmask16 test_mm512_cmpord_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmpord_ps_mask
  // CHECK: fcmp ord <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmpord_ps_mask(a, b);
}

__mmask8 test_mm512_mask_cmpord_pd_mask(__mmask8 k, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmpord_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp ord <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpord_pd_mask(k, a, b);
}

__mmask16 test_mm512_mask_cmpord_ps_mask(__mmask16 k, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmpord_ps_mask
  // CHECK: [[CMP:%.*]] = fcmp ord <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpord_ps_mask(k, a, b);
}

__mmask8 test_mm512_cmpunord_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_cmpunord_pd_mask
  // CHECK: fcmp uno <8 x double> %{{.*}}, %{{.*}}
  return _mm512_cmpunord_pd_mask(a, b);
}

__mmask16 test_mm512_cmpunord_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_cmpunord_ps_mask
  // CHECK: fcmp uno <16 x float> %{{.*}}, %{{.*}}
  return _mm512_cmpunord_ps_mask(a, b);
}

__mmask8 test_mm512_mask_cmpunord_pd_mask(__mmask8 k, __m512d a, __m512d b) {
  // CHECK-LABEL: test_mm512_mask_cmpunord_pd_mask
  // CHECK: [[CMP:%.*]] = fcmp uno <8 x double> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpunord_pd_mask(k, a, b);
}

__mmask16 test_mm512_mask_cmpunord_ps_mask(__mmask16 k, __m512 a, __m512 b) {
  // CHECK-LABEL: test_mm512_mask_cmpunord_ps_mask
  // CHECK: [[CMP:%.*]] = fcmp uno <16 x float> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmpunord_ps_mask(k, a, b);
}

__m256d test_mm512_extractf64x4_pd(__m512d a)
{
  // CHECK-LABEL: test_mm512_extractf64x4_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm512_extractf64x4_pd(a, 1);
}

__m256d test_mm512_mask_extractf64x4_pd(__m256d  __W,__mmask8  __U,__m512d __A){
  // CHECK-LABEL: test_mm512_mask_extractf64x4_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm512_mask_extractf64x4_pd( __W, __U, __A, 1);
}

__m256d test_mm512_maskz_extractf64x4_pd(__mmask8  __U,__m512d __A){
  // CHECK-LABEL: test_mm512_maskz_extractf64x4_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm512_maskz_extractf64x4_pd( __U, __A, 1);
}

__m128 test_mm512_extractf32x4_ps(__m512 a)
{
  // CHECK-LABEL: test_mm512_extractf32x4_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm512_extractf32x4_ps(a, 1);
}

__m128 test_mm512_mask_extractf32x4_ps(__m128 __W, __mmask8  __U,__m512 __A){
  // CHECK-LABEL: test_mm512_mask_extractf32x4_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm512_mask_extractf32x4_ps( __W, __U, __A, 1);
}

__m128 test_mm512_maskz_extractf32x4_ps( __mmask8  __U,__m512 __A){
  // CHECK-LABEL: test_mm512_maskz_extractf32x4_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm512_maskz_extractf32x4_ps(__U, __A, 1);
}

__mmask16 test_mm512_cmpeq_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpeq_epu32_mask
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmpeq_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpeq_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpeq_epu32_mask
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmpeq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpeq_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpeq_epu64_mask
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmpeq_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpeq_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpeq_epu64_mask
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmpeq_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpge_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpge_epi32_mask
  // CHECK: icmp sge <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmpge_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpge_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpge_epi32_mask
  // CHECK: icmp sge <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpge_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpge_epi64_mask
  // CHECK: icmp sge <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpge_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpge_epi64_mask
  // CHECK: icmp sge <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpge_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpge_epu32_mask
  // CHECK: icmp uge <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmpge_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpge_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpge_epu32_mask
  // CHECK: icmp uge <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpge_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpge_epu64_mask
  // CHECK: icmp uge <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpge_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpge_epu64_mask
  // CHECK: icmp uge <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpgt_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpgt_epu32_mask
  // CHECK: icmp ugt <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmpgt_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpgt_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpgt_epu32_mask
  // CHECK: icmp ugt <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpgt_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpgt_epu64_mask
  // CHECK: icmp ugt <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpgt_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpgt_epu64_mask
  // CHECK: icmp ugt <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmple_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmple_epi32_mask
  // CHECK: icmp sle <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmple_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmple_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmple_epi32_mask
  // CHECK: icmp sle <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmple_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmple_epi64_mask
  // CHECK: icmp sle <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmple_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmple_epi64_mask
  // CHECK: icmp sle <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmple_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmple_epu32_mask
  // CHECK: icmp ule <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmple_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmple_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmple_epu32_mask
  // CHECK: icmp ule <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmple_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmple_epu64_mask
  // CHECK: icmp ule <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmple_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmple_epu64_mask
  // CHECK: icmp ule <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmplt_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmplt_epi32_mask
  // CHECK: icmp slt <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmplt_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmplt_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmplt_epi32_mask
  // CHECK: icmp slt <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmplt_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmplt_epi64_mask
  // CHECK: icmp slt <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmplt_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmplt_epi64_mask
  // CHECK: icmp slt <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmplt_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmplt_epu32_mask
  // CHECK: icmp ult <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmplt_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmplt_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmplt_epu32_mask
  // CHECK: icmp ult <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmplt_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmplt_epu64_mask
  // CHECK: icmp ult <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmplt_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmplt_epu64_mask
  // CHECK: icmp ult <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpneq_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpneq_epi32_mask
  // CHECK: icmp ne <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmpneq_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpneq_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpneq_epi32_mask
  // CHECK: icmp ne <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpneq_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpneq_epi64_mask
  // CHECK: icmp ne <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpneq_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpneq_epi64_mask
  // CHECK: icmp ne <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpneq_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpneq_epu32_mask
  // CHECK: icmp ne <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmpneq_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpneq_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpneq_epu32_mask
  // CHECK: icmp ne <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpneq_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmpneq_epu64_mask
  // CHECK: icmp ne <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpneq_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmpneq_epu64_mask
  // CHECK: icmp ne <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmp_eq_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmp_eq_epi32_mask
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmp_epi32_mask(__a, __b, _MM_CMPINT_EQ);
}

__mmask16 test_mm512_mask_cmp_eq_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmp_eq_epi32_mask
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmp_epi32_mask(__u, __a, __b, _MM_CMPINT_EQ);
}

__mmask8 test_mm512_cmp_eq_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmp_eq_epi64_mask
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmp_epi64_mask(__a, __b, _MM_CMPINT_EQ);
}

__mmask8 test_mm512_mask_cmp_eq_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmp_eq_epi64_mask
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmp_epi64_mask(__u, __a, __b, _MM_CMPINT_EQ);
}

__mmask16 test_mm512_cmp_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmp_epu32_mask
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_cmp_epu32_mask(__a, __b, 0);
}

__mmask16 test_mm512_mask_cmp_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmp_epu32_mask
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm512_mask_cmp_epu32_mask(__u, __a, __b, 0);
}

__mmask8 test_mm512_cmp_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_cmp_epu64_mask
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_cmp_epu64_mask(__a, __b, 0);
}

__mmask8 test_mm512_mask_cmp_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_cmp_epu64_mask
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm512_mask_cmp_epu64_mask(__u, __a, __b, 0);
}

__m512i test_mm512_mask_and_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_and_epi32
  // CHECK: and <16 x i32> 
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_and_epi32(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_and_epi32(__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_maskz_and_epi32
  // CHECK: and <16 x i32> 
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_and_epi32(__k,__a, __b);
}

__m512i test_mm512_mask_and_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_and_epi64
  // CHECK: %[[AND_RES:.*]] = and <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x i64> %[[AND_RES]], <8 x i64> %{{.*}}
  return _mm512_mask_and_epi64(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_and_epi64(__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_maskz_and_epi64
  // CHECK: %[[AND_RES:.*]] = and <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x i64> %[[AND_RES]], <8 x i64> %{{.*}}
  return _mm512_maskz_and_epi64(__k,__a, __b);
}

__m512i test_mm512_mask_or_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_or_epi32
  // CHECK: or <16 x i32> 
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_or_epi32(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_or_epi32(__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_maskz_or_epi32
  // CHECK: or <16 x i32> 
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_or_epi32(__k,__a, __b);
}

__m512i test_mm512_mask_or_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_or_epi64
  // CHECK: %[[OR_RES:.*]] = or <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x i64> %[[OR_RES]], <8 x i64> %{{.*}}
  return _mm512_mask_or_epi64(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_or_epi64(__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_maskz_or_epi64
  // CHECK: %[[OR_RES:.*]] = or <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x i64> %[[OR_RES]], <8 x i64> %{{.*}}
  return _mm512_maskz_or_epi64(__k,__a, __b);
}

__m512i test_mm512_mask_xor_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_xor_epi32
  // CHECK: xor <16 x i32> 
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_xor_epi32(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_xor_epi32(__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_maskz_xor_epi32
  // CHECK: xor <16 x i32> 
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_xor_epi32(__k,__a, __b);
}

__m512i test_mm512_mask_xor_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_mask_xor_epi64
  // CHECK: %[[XOR_RES:.*]] = xor <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x i64> %[[XOR_RES]], <8 x i64> %{{.*}}
  return _mm512_mask_xor_epi64(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_xor_epi64(__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_maskz_xor_epi64
  // CHECK: %[[XOR_RES:.*]] = xor <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x i64> %[[XOR_RES]], <8 x i64> %{{.*}}
  return _mm512_maskz_xor_epi64(__k,__a, __b);
}

__m512i test_mm512_and_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_and_epi32
  // CHECK: and <16 x i32>
  return _mm512_and_epi32(__a, __b);
}

__m512i test_mm512_and_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_and_epi64
  // CHECK: and <8 x i64>
  return _mm512_and_epi64(__a, __b);
}

__m512i test_mm512_or_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_or_epi32
  // CHECK: or <16 x i32>
  return _mm512_or_epi32(__a, __b);
}

__m512i test_mm512_or_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_or_epi64
  // CHECK: or <8 x i64>
  return _mm512_or_epi64(__a, __b);
}

__m512i test_mm512_xor_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_xor_epi32
  // CHECK: xor <16 x i32>
  return _mm512_xor_epi32(__a, __b);
}

__m512i test_mm512_xor_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: test_mm512_xor_epi64
  // CHECK: xor <8 x i64>
  return _mm512_xor_epi64(__a, __b);
}

__m512i test_mm512_maskz_andnot_epi32 (__mmask16 __k,__m512i __A, __m512i __B){
  // CHECK-LABEL: test_mm512_maskz_andnot_epi32
  // CHECK: xor <16 x i32> %{{.*}}, splat (i32 -1)
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_andnot_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_andnot_epi32 (__mmask16 __k,__m512i __A, __m512i __B,
                                      __m512i __src) {
  // CHECK-LABEL: test_mm512_mask_andnot_epi32
  // CHECK: xor <16 x i32> %{{.*}}, splat (i32 -1)
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_andnot_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_andnot_si512(__m512i __A, __m512i __B)
{
  //CHECK-LABEL: test_mm512_andnot_si512
  //CHECK: load {{.*}}%__A.addr.i, align 64
  //CHECK: %not.i = xor{{.*}}, splat (i64 -1)
  //CHECK: load {{.*}}%__B.addr.i, align 64
  //CHECK: and <8 x i64> %not.i,{{.*}}

  return _mm512_andnot_si512(__A, __B);
}

__m512i test_mm512_andnot_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_andnot_epi32
  // CHECK: xor <16 x i32> %{{.*}}, splat (i32 -1)
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  return _mm512_andnot_epi32(__A,__B);
}

__m512i test_mm512_maskz_andnot_epi64 (__mmask8 __k,__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_andnot_epi64
  // CHECK: xor <8 x i64> %{{.*}}, splat (i64 -1)
  // CHECK: and <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_andnot_epi64(__k,__A,__B);
}

__m512i test_mm512_mask_andnot_epi64 (__mmask8 __k,__m512i __A, __m512i __B, 
                                      __m512i __src) {
  //CHECK-LABEL: test_mm512_mask_andnot_epi64
  // CHECK: xor <8 x i64> %{{.*}}, splat (i64 -1)
  // CHECK: and <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_andnot_epi64(__src,__k,__A,__B);
}

__m512i test_mm512_andnot_epi64(__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_andnot_epi64
  // CHECK: xor <8 x i64> %{{.*}}, splat (i64 -1)
  // CHECK: and <8 x i64> %{{.*}}, %{{.*}}
  return _mm512_andnot_epi64(__A,__B);
}

__m512i test_mm512_maskz_sub_epi32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_maskz_sub_epi32
  //CHECK: sub <16 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_sub_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_sub_epi32 (__mmask16 __k,__m512i __A, __m512i __B, 
                                   __m512i __src) {
  //CHECK-LABEL: test_mm512_mask_sub_epi32
  //CHECK: sub <16 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_sub_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_sub_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_sub_epi32
  //CHECK: sub <16 x i32>
  return _mm512_sub_epi32(__A,__B);
}

__m512i test_mm512_maskz_sub_epi64 (__mmask8 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_maskz_sub_epi64
  //CHECK: sub <8 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_sub_epi64(__k,__A,__B);
}

__m512i test_mm512_mask_sub_epi64 (__mmask8 __k,__m512i __A, __m512i __B, 
                                   __m512i __src) {
  //CHECK-LABEL: test_mm512_mask_sub_epi64
  //CHECK: sub <8 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_sub_epi64(__src,__k,__A,__B);
}

__m512i test_mm512_sub_epi64(__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_sub_epi64
  //CHECK: sub <8 x i64>
  return _mm512_sub_epi64(__A,__B);
}

__m512i test_mm512_maskz_add_epi32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_maskz_add_epi32
  //CHECK: add <16 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_add_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_add_epi32 (__mmask16 __k,__m512i __A, __m512i __B, 
                                   __m512i __src) {
  //CHECK-LABEL: test_mm512_mask_add_epi32
  //CHECK: add <16 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_add_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_add_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_add_epi32
  //CHECK: add <16 x i32>
  return _mm512_add_epi32(__A,__B);
}

__m512i test_mm512_maskz_add_epi64 (__mmask8 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_maskz_add_epi64
  //CHECK: add <8 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_add_epi64(__k,__A,__B);
}

__m512i test_mm512_mask_add_epi64 (__mmask8 __k,__m512i __A, __m512i __B, 
                                   __m512i __src) {
  //CHECK-LABEL: test_mm512_mask_add_epi64
  //CHECK: add <8 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_add_epi64(__src,__k,__A,__B);
}

__m512i test_mm512_add_epi64(__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_add_epi64
  //CHECK: add <8 x i64>
  return _mm512_add_epi64(__A,__B);
}

__m512i test_mm512_mul_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_mul_epi32
  //CHECK: shl <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: ashr <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: shl <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: ashr <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: mul <8 x i64> %{{.*}}, %{{.*}}
  return _mm512_mul_epi32(__A,__B);
}

__m512i test_mm512_maskz_mul_epi32 (__mmask8 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_maskz_mul_epi32
  //CHECK: shl <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: ashr <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: shl <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: ashr <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: mul <8 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_mul_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_mul_epi32 (__mmask8 __k,__m512i __A, __m512i __B, __m512i __src) {
  //CHECK-LABEL: test_mm512_mask_mul_epi32
  //CHECK: shl <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: ashr <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: shl <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: ashr <8 x i64> %{{.*}}, splat (i64 32)
  //CHECK: mul <8 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_mul_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_mul_epu32 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_mul_epu32
  //CHECK: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  //CHECK: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  //CHECK: mul <8 x i64> %{{.*}}, %{{.*}}
  return _mm512_mul_epu32(__A,__B);
}

__m512i test_mm512_maskz_mul_epu32 (__mmask8 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_maskz_mul_epu32
  //CHECK: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  //CHECK: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  //CHECK: mul <8 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_mul_epu32(__k,__A,__B);
}

__m512i test_mm512_mask_mul_epu32 (__mmask8 __k,__m512i __A, __m512i __B, __m512i __src) {
  //CHECK-LABEL: test_mm512_mask_mul_epu32
  //CHECK: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  //CHECK: and <8 x i64> %{{.*}}, splat (i64 4294967295)
  //CHECK: mul <8 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_mul_epu32(__src,__k,__A,__B);
}

__m512i test_mm512_maskz_mullo_epi32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_maskz_mullo_epi32
  //CHECK: mul <16 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_mullo_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_mullo_epi32 (__mmask16 __k,__m512i __A, __m512i __B, __m512i __src) {
  //CHECK-LABEL: test_mm512_mask_mullo_epi32
  //CHECK: mul <16 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_mullo_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_mullo_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: test_mm512_mullo_epi32
  //CHECK: mul <16 x i32>
  return _mm512_mullo_epi32(__A,__B);
}

__m512i test_mm512_mullox_epi64 (__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mullox_epi64
  // CHECK: mul <8 x i64>
  return (__m512i) _mm512_mullox_epi64(__A, __B);
}

__m512i test_mm512_mask_mullox_epi64 (__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_mullox_epi64
  // CHECK: mul <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return (__m512i) _mm512_mask_mullox_epi64(__W, __U, __A, __B);
}

__m512d test_mm512_add_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_add_round_pd
  // CHECK: @llvm.x86.avx512.add.pd.512
  return _mm512_add_round_pd(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_add_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_add_round_pd
  // CHECK: @llvm.x86.avx512.add.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_add_round_pd(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_add_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_add_round_pd
  // CHECK: @llvm.x86.avx512.add.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_add_round_pd(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_add_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_add_pd
  // CHECK: fadd <8 x double> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_add_pd(__W,__U,__A,__B); 
}
__m512d test_mm512_maskz_add_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_add_pd
  // CHECK: fadd <8 x double> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_add_pd(__U,__A,__B); 
}
__m512 test_mm512_add_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_add_round_ps
  // CHECK: @llvm.x86.avx512.add.ps.512
  return _mm512_add_round_ps(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_add_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_add_round_ps
  // CHECK: @llvm.x86.avx512.add.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_add_round_ps(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_add_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_add_round_ps
  // CHECK: @llvm.x86.avx512.add.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_add_round_ps(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_add_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_add_ps
  // CHECK: fadd <16 x float> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_add_ps(__W,__U,__A,__B); 
}
__m512 test_mm512_maskz_add_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_add_ps
  // CHECK: fadd <16 x float> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_add_ps(__U,__A,__B); 
}
__m128 test_mm_add_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_add_round_ss
  // CHECK: @llvm.x86.avx512.mask.add.ss.round
  return _mm_add_round_ss(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_mask_add_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_add_round_ss
  // CHECK: @llvm.x86.avx512.mask.add.ss.round
  return _mm_mask_add_round_ss(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_maskz_add_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_add_round_ss
  // CHECK: @llvm.x86.avx512.mask.add.ss.round
  return _mm_maskz_add_round_ss(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_mask_add_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_add_ss
  // CHECK-NOT: @llvm.x86.avx512.mask.add.ss.round
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fadd float %{{.*}}, %{{.*}}
  // CHECK: insertelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, float %{{.*}}, float %{{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_mask_add_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_add_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_add_ss
  // CHECK-NOT: @llvm.x86.avx512.mask.add.ss.round
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fadd float %{{.*}}, %{{.*}}
  // CHECK: insertelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, float %{{.*}}, float %{{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_maskz_add_ss(__U,__A,__B); 
}
__m128d test_mm_add_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_add_round_sd
  // CHECK: @llvm.x86.avx512.mask.add.sd.round
  return _mm_add_round_sd(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_mask_add_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_add_round_sd
  // CHECK: @llvm.x86.avx512.mask.add.sd.round
  return _mm_mask_add_round_sd(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_maskz_add_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_add_round_sd
  // CHECK: @llvm.x86.avx512.mask.add.sd.round
  return _mm_maskz_add_round_sd(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_mask_add_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_add_sd
  // CHECK-NOT: @llvm.x86.avx512.mask.add.sd.round
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fadd double %{{.*}}, %{{.*}}
  // CHECK: insertelement <2 x double> {{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, double %{{.*}}, double %{{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_mask_add_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_add_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_add_sd
  // CHECK-NOT: @llvm.x86.avx512.mask.add.sd.round
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fadd double %{{.*}}, %{{.*}}
  // CHECK: insertelement <2 x double> {{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, double %{{.*}}, double %{{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_maskz_add_sd(__U,__A,__B); 
}
__m512d test_mm512_sub_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_sub_round_pd
  // CHECK: @llvm.x86.avx512.sub.pd.512
  return _mm512_sub_round_pd(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_sub_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_sub_round_pd
  // CHECK: @llvm.x86.avx512.sub.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_sub_round_pd(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_sub_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_sub_round_pd
  // CHECK: @llvm.x86.avx512.sub.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_sub_round_pd(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_sub_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_sub_pd
  // CHECK: fsub <8 x double> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_sub_pd(__W,__U,__A,__B); 
}
__m512d test_mm512_maskz_sub_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_sub_pd
  // CHECK: fsub <8 x double> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_sub_pd(__U,__A,__B); 
}
__m512 test_mm512_sub_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_sub_round_ps
  // CHECK: @llvm.x86.avx512.sub.ps.512
  return _mm512_sub_round_ps(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_sub_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_sub_round_ps
  // CHECK: @llvm.x86.avx512.sub.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_sub_round_ps(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_sub_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_sub_round_ps
  // CHECK: @llvm.x86.avx512.sub.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_sub_round_ps(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_sub_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_sub_ps
  // CHECK: fsub <16 x float> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_sub_ps(__W,__U,__A,__B); 
}
__m512 test_mm512_maskz_sub_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_sub_ps
  // CHECK: fsub <16 x float> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_sub_ps(__U,__A,__B); 
}
__m128 test_mm_sub_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_sub_round_ss
  // CHECK: @llvm.x86.avx512.mask.sub.ss.round
  return _mm_sub_round_ss(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_mask_sub_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_sub_round_ss
  // CHECK: @llvm.x86.avx512.mask.sub.ss.round
  return _mm_mask_sub_round_ss(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_maskz_sub_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_sub_round_ss
  // CHECK: @llvm.x86.avx512.mask.sub.ss.round
  return _mm_maskz_sub_round_ss(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_mask_sub_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_sub_ss
  // CHECK-NOT: @llvm.x86.avx512.mask.sub.ss.round
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fsub float %{{.*}}, %{{.*}}
  // CHECK: insertelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, float %{{.*}}, float %{{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_mask_sub_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_sub_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_sub_ss
  // CHECK-NOT: @llvm.x86.avx512.mask.sub.ss.round
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fsub float %{{.*}}, %{{.*}}
  // CHECK: insertelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, float %{{.*}}, float %{{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_maskz_sub_ss(__U,__A,__B); 
}
__m128d test_mm_sub_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_sub_round_sd
  // CHECK: @llvm.x86.avx512.mask.sub.sd.round
  return _mm_sub_round_sd(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_mask_sub_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_sub_round_sd
  // CHECK: @llvm.x86.avx512.mask.sub.sd.round
  return _mm_mask_sub_round_sd(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_maskz_sub_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_sub_round_sd
  // CHECK: @llvm.x86.avx512.mask.sub.sd.round
  return _mm_maskz_sub_round_sd(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_mask_sub_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_sub_sd
  // CHECK-NOT: @llvm.x86.avx512.mask.sub.sd.round
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fsub double %{{.*}}, %{{.*}}
  // CHECK: insertelement <2 x double> {{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, double %{{.*}}, double %{{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_mask_sub_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_sub_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_sub_sd
  // CHECK-NOT: @llvm.x86.avx512.mask.sub.sd.round
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fsub double %{{.*}}, %{{.*}}
  // CHECK: insertelement <2 x double> {{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, double %{{.*}}, double %{{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_maskz_sub_sd(__U,__A,__B); 
}
__m512d test_mm512_mul_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mul_round_pd
  // CHECK: @llvm.x86.avx512.mul.pd.512
  return _mm512_mul_round_pd(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_mul_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_mul_round_pd
  // CHECK: @llvm.x86.avx512.mul.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_mul_round_pd(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_mul_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_mul_round_pd
  // CHECK: @llvm.x86.avx512.mul.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_mul_round_pd(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_mul_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_mul_pd
  // CHECK: fmul <8 x double> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_mul_pd(__W,__U,__A,__B); 
}
__m512d test_mm512_maskz_mul_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_mul_pd
  // CHECK: fmul <8 x double> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_mul_pd(__U,__A,__B); 
}
__m512 test_mm512_mul_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mul_round_ps
  // CHECK: @llvm.x86.avx512.mul.ps.512
  return _mm512_mul_round_ps(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_mul_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_mul_round_ps
  // CHECK: @llvm.x86.avx512.mul.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_mul_round_ps(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_mul_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_mul_round_ps
  // CHECK: @llvm.x86.avx512.mul.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_mul_round_ps(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_mul_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_mul_ps
  // CHECK: fmul <16 x float> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_mul_ps(__W,__U,__A,__B); 
}
__m512 test_mm512_maskz_mul_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_mul_ps
  // CHECK: fmul <16 x float> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_mul_ps(__U,__A,__B); 
}
__m128 test_mm_mul_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mul_round_ss
  // CHECK: @llvm.x86.avx512.mask.mul.ss.round
  return _mm_mul_round_ss(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_mask_mul_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_mul_round_ss
  // CHECK: @llvm.x86.avx512.mask.mul.ss.round
  return _mm_mask_mul_round_ss(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_maskz_mul_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_mul_round_ss
  // CHECK: @llvm.x86.avx512.mask.mul.ss.round
  return _mm_maskz_mul_round_ss(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_mask_mul_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_mul_ss
  // CHECK-NOT: @llvm.x86.avx512.mask.mul.ss.round
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fmul float %{{.*}}, %{{.*}}
  // CHECK: insertelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, float %{{.*}}, float %{{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_mask_mul_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_mul_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_mul_ss
  // CHECK-NOT: @llvm.x86.avx512.mask.mul.ss.round
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fmul float %{{.*}}, %{{.*}}
  // CHECK: insertelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, float %{{.*}}, float %{{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_maskz_mul_ss(__U,__A,__B); 
}
__m128d test_mm_mul_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mul_round_sd
  // CHECK: @llvm.x86.avx512.mask.mul.sd.round
  return _mm_mul_round_sd(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_mask_mul_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_mul_round_sd
  // CHECK: @llvm.x86.avx512.mask.mul.sd.round
  return _mm_mask_mul_round_sd(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_maskz_mul_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_mul_round_sd
  // CHECK: @llvm.x86.avx512.mask.mul.sd.round
  return _mm_maskz_mul_round_sd(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_mask_mul_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_mul_sd
  // CHECK-NOT: @llvm.x86.avx512.mask.mul.sd.round
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fmul double %{{.*}}, %{{.*}}
  // CHECK: insertelement <2 x double> {{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, double %{{.*}}, double %{{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_mask_mul_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_mul_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_mul_sd
  // CHECK-NOT: @llvm.x86.avx512.mask.mul.sd.round
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fmul double %{{.*}}, %{{.*}}
  // CHECK: insertelement <2 x double> {{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, double %{{.*}}, double %{{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_maskz_mul_sd(__U,__A,__B); 
}
__m512d test_mm512_div_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_div_round_pd
  // CHECK: @llvm.x86.avx512.div.pd.512
  return _mm512_div_round_pd(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_mask_div_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_div_round_pd
  // CHECK: @llvm.x86.avx512.div.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_div_round_pd(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_maskz_div_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_div_round_pd
  // CHECK: @llvm.x86.avx512.div.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_div_round_pd(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512d test_mm512_div_pd(__m512d __a, __m512d __b) {
  // CHECK-LABEL: test_mm512_div_pd
  // CHECK: fdiv <8 x double>
  return _mm512_div_pd(__a,__b); 
}
__m512d test_mm512_mask_div_pd(__m512d __w, __mmask8 __u, __m512d __a, __m512d __b) {
  // CHECK-LABEL: test_mm512_mask_div_pd
  // CHECK: fdiv <8 x double> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_div_pd(__w,__u,__a,__b); 
}
__m512d test_mm512_maskz_div_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_div_pd
  // CHECK: fdiv <8 x double> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_div_pd(__U,__A,__B); 
}
__m512 test_mm512_div_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_div_round_ps
  // CHECK: @llvm.x86.avx512.div.ps.512
  return _mm512_div_round_ps(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_mask_div_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_div_round_ps
  // CHECK: @llvm.x86.avx512.div.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_div_round_ps(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_maskz_div_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_div_round_ps
  // CHECK: @llvm.x86.avx512.div.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_div_round_ps(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512 test_mm512_div_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_div_ps
  // CHECK: fdiv <16 x float>
  return _mm512_div_ps(__A,__B); 
}
__m512 test_mm512_mask_div_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_div_ps
  // CHECK: fdiv <16 x float> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_div_ps(__W,__U,__A,__B); 
}
__m512 test_mm512_maskz_div_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_div_ps
  // CHECK: fdiv <16 x float> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_div_ps(__U,__A,__B); 
}
__m128 test_mm_div_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_div_round_ss
  // CHECK: @llvm.x86.avx512.mask.div.ss.round
  return _mm_div_round_ss(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_mask_div_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_div_round_ss
  // CHECK: @llvm.x86.avx512.mask.div.ss.round
  return _mm_mask_div_round_ss(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_maskz_div_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_div_round_ss
  // CHECK: @llvm.x86.avx512.mask.div.ss.round
  return _mm_maskz_div_round_ss(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128 test_mm_mask_div_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_div_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fdiv float %{{.*}}, %{{.*}}
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, float %{{.*}}, float %{{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_mask_div_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_div_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_div_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fdiv float %{{.*}}, %{{.*}}
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, float %{{.*}}, float %{{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_maskz_div_ss(__U,__A,__B); 
}
__m128d test_mm_div_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_div_round_sd
  // CHECK: @llvm.x86.avx512.mask.div.sd.round
  return _mm_div_round_sd(__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_mask_div_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_div_round_sd
  // CHECK: @llvm.x86.avx512.mask.div.sd.round
  return _mm_mask_div_round_sd(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_maskz_div_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_div_round_sd
  // CHECK: @llvm.x86.avx512.mask.div.sd.round
  return _mm_maskz_div_round_sd(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m128d test_mm_mask_div_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_div_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fdiv double %{{.*}}, %{{.*}}
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, double %{{.*}}, double %{{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_mask_div_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_div_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_div_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: fdiv double %{{.*}}, %{{.*}}
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 %{{.*}}, double %{{.*}}, double %{{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_maskz_div_sd(__U,__A,__B); 
}
__m128 test_mm_max_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_max_round_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_max_round_ss(__A,__B,0x08); 
}
__m128 test_mm_mask_max_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_max_round_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_mask_max_round_ss(__W,__U,__A,__B,0x08); 
}
__m128 test_mm_maskz_max_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_max_round_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_maskz_max_round_ss(__U,__A,__B,0x08); 
}
__m128 test_mm_mask_max_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_max_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_mask_max_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_max_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_max_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_maskz_max_ss(__U,__A,__B); 
}
__m128d test_mm_max_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_max_round_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_max_round_sd(__A,__B,0x08); 
}
__m128d test_mm_mask_max_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_max_round_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_mask_max_round_sd(__W,__U,__A,__B,0x08); 
}
__m128d test_mm_maskz_max_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_max_round_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_maskz_max_round_sd(__U,__A,__B,0x08); 
}
__m128d test_mm_mask_max_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_max_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_mask_max_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_max_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_max_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_maskz_max_sd(__U,__A,__B); 
}
__m128 test_mm_min_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_min_round_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_min_round_ss(__A,__B,0x08); 
}
__m128 test_mm_mask_min_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_min_round_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_mask_min_round_ss(__W,__U,__A,__B,0x08); 
}
__m128 test_mm_maskz_min_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_min_round_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_maskz_min_round_ss(__U,__A,__B,0x08); 
}
__m128 test_mm_mask_min_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_min_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_mask_min_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_min_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_min_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_maskz_min_ss(__U,__A,__B); 
}
__m128d test_mm_min_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_min_round_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_min_round_sd(__A,__B,0x08); 
}
__m128d test_mm_mask_min_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_min_round_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_mask_min_round_sd(__W,__U,__A,__B,0x08); 
}
__m128d test_mm_maskz_min_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_min_round_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_maskz_min_round_sd(__U,__A,__B,0x08); 
}
__m128d test_mm_mask_min_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_min_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_mask_min_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_min_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_min_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_maskz_min_sd(__U,__A,__B); 
}

__m512 test_mm512_undefined(void) {
  // CHECK-LABEL: test_mm512_undefined
  // CHECK: ret <16 x float> zeroinitializer
  return _mm512_undefined();
}

__m512 test_mm512_undefined_ps(void) {
  // CHECK-LABEL: test_mm512_undefined_ps
  // CHECK: ret <16 x float> zeroinitializer
  return _mm512_undefined_ps();
}

__m512d test_mm512_undefined_pd(void) {
  // CHECK-LABEL: test_mm512_undefined_pd
  // CHECK: ret <8 x double> zeroinitializer
  return _mm512_undefined_pd();
}

__m512i test_mm512_undefined_epi32(void) {
  // CHECK-LABEL: test_mm512_undefined_epi32
  // CHECK: ret <8 x i64> zeroinitializer
  return _mm512_undefined_epi32();
}

__m512i test_mm512_cvtepi8_epi32(__m128i __A) {
  // CHECK-LABEL: test_mm512_cvtepi8_epi32
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i32>
  return _mm512_cvtepi8_epi32(__A); 
}

__m512i test_mm512_mask_cvtepi8_epi32(__m512i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi8_epi32
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_cvtepi8_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepi8_epi32(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi8_epi32
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_cvtepi8_epi32(__U, __A); 
}

__m512i test_mm512_cvtepi8_epi64(__m128i __A) {
  // CHECK-LABEL: test_mm512_cvtepi8_epi64
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i64>
  return _mm512_cvtepi8_epi64(__A); 
}

__m512i test_mm512_mask_cvtepi8_epi64(__m512i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi8_epi64
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_cvtepi8_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepi8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi8_epi64
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_cvtepi8_epi64(__U, __A); 
}

__m512i test_mm512_cvtepi32_epi64(__m256i __X) {
  // CHECK-LABEL: test_mm512_cvtepi32_epi64
  // CHECK: sext <8 x i32> %{{.*}} to <8 x i64>
  return _mm512_cvtepi32_epi64(__X); 
}

__m512i test_mm512_mask_cvtepi32_epi64(__m512i __W, __mmask8 __U, __m256i __X) {
  // CHECK-LABEL: test_mm512_mask_cvtepi32_epi64
  // CHECK: sext <8 x i32> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_cvtepi32_epi64(__W, __U, __X); 
}

__m512i test_mm512_maskz_cvtepi32_epi64(__mmask8 __U, __m256i __X) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi32_epi64
  // CHECK: sext <8 x i32> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_cvtepi32_epi64(__U, __X); 
}

__m512i test_mm512_cvtepi16_epi32(__m256i __A) {
  // CHECK-LABEL: test_mm512_cvtepi16_epi32
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  return _mm512_cvtepi16_epi32(__A); 
}

__m512i test_mm512_mask_cvtepi16_epi32(__m512i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi16_epi32
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_cvtepi16_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepi16_epi32(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi16_epi32
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_cvtepi16_epi32(__U, __A); 
}

__m512i test_mm512_cvtepi16_epi64(__m128i __A) {
  // CHECK-LABEL: test_mm512_cvtepi16_epi64
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i64>
  return _mm512_cvtepi16_epi64(__A); 
}

__m512i test_mm512_mask_cvtepi16_epi64(__m512i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi16_epi64
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_cvtepi16_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepi16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi16_epi64
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_cvtepi16_epi64(__U, __A); 
}

__m512i test_mm512_cvtepu8_epi32(__m128i __A) {
  // CHECK-LABEL: test_mm512_cvtepu8_epi32
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i32>
  return _mm512_cvtepu8_epi32(__A); 
}

__m512i test_mm512_mask_cvtepu8_epi32(__m512i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepu8_epi32
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_cvtepu8_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepu8_epi32(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu8_epi32
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_cvtepu8_epi32(__U, __A); 
}

__m512i test_mm512_cvtepu8_epi64(__m128i __A) {
  // CHECK-LABEL: test_mm512_cvtepu8_epi64
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i64>
  return _mm512_cvtepu8_epi64(__A); 
}

__m512i test_mm512_mask_cvtepu8_epi64(__m512i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepu8_epi64
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_cvtepu8_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepu8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu8_epi64
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_cvtepu8_epi64(__U, __A); 
}

__m512i test_mm512_cvtepu32_epi64(__m256i __X) {
  // CHECK-LABEL: test_mm512_cvtepu32_epi64
  // CHECK: zext <8 x i32> %{{.*}} to <8 x i64>
  return _mm512_cvtepu32_epi64(__X); 
}

__m512i test_mm512_mask_cvtepu32_epi64(__m512i __W, __mmask8 __U, __m256i __X) {
  // CHECK-LABEL: test_mm512_mask_cvtepu32_epi64
  // CHECK: zext <8 x i32> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_cvtepu32_epi64(__W, __U, __X); 
}

__m512i test_mm512_maskz_cvtepu32_epi64(__mmask8 __U, __m256i __X) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu32_epi64
  // CHECK: zext <8 x i32> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_cvtepu32_epi64(__U, __X); 
}

__m512i test_mm512_cvtepu16_epi32(__m256i __A) {
  // CHECK-LABEL: test_mm512_cvtepu16_epi32
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  return _mm512_cvtepu16_epi32(__A); 
}

__m512i test_mm512_mask_cvtepu16_epi32(__m512i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepu16_epi32
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_cvtepu16_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepu16_epi32(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu16_epi32
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_cvtepu16_epi32(__U, __A); 
}

__m512i test_mm512_cvtepu16_epi64(__m128i __A) {
  // CHECK-LABEL: test_mm512_cvtepu16_epi64
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i64>
  return _mm512_cvtepu16_epi64(__A); 
}

__m512i test_mm512_mask_cvtepu16_epi64(__m512i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepu16_epi64
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_cvtepu16_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepu16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu16_epi64
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i64>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_cvtepu16_epi64(__U, __A); 
}


__m512i test_mm512_rol_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_rol_epi32
  // CHECK: @llvm.fshl.v16i32
  return _mm512_rol_epi32(__A, 5); 
}

__m512i test_mm512_mask_rol_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_rol_epi32
  // CHECK: @llvm.fshl.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_rol_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_rol_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_rol_epi32
  // CHECK: @llvm.fshl.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_rol_epi32(__U, __A, 5); 
}

__m512i test_mm512_rol_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_rol_epi64
  // CHECK: @llvm.fshl.v8i64
  return _mm512_rol_epi64(__A, 5); 
}

__m512i test_mm512_mask_rol_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_rol_epi64
  // CHECK: @llvm.fshl.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_rol_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_rol_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_rol_epi64
  // CHECK: @llvm.fshl.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_rol_epi64(__U, __A, 5); 
}

__m512i test_mm512_rolv_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_rolv_epi32
  // CHECK: @llvm.fshl.v16i32
  return _mm512_rolv_epi32(__A, __B); 
}

__m512i test_mm512_mask_rolv_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_rolv_epi32
  // CHECK: @llvm.fshl.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_rolv_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_rolv_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_rolv_epi32
  // CHECK: @llvm.fshl.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_rolv_epi32(__U, __A, __B); 
}

__m512i test_mm512_rolv_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_rolv_epi64
  // CHECK: @llvm.fshl.v8i64
  return _mm512_rolv_epi64(__A, __B); 
}

__m512i test_mm512_mask_rolv_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_rolv_epi64
  // CHECK: @llvm.fshl.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_rolv_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_rolv_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_rolv_epi64
  // CHECK: @llvm.fshl.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_rolv_epi64(__U, __A, __B); 
}

__m512i test_mm512_ror_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_ror_epi32
  // CHECK: @llvm.fshr.v16i32
  return _mm512_ror_epi32(__A, 5); 
}

__m512i test_mm512_mask_ror_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_ror_epi32
  // CHECK: @llvm.fshr.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_ror_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_ror_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_ror_epi32
  // CHECK: @llvm.fshr.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_ror_epi32(__U, __A, 5); 
}

__m512i test_mm512_ror_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_ror_epi64
  // CHECK: @llvm.fshr.v8i64
  return _mm512_ror_epi64(__A, 5); 
}

__m512i test_mm512_mask_ror_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_ror_epi64
  // CHECK: @llvm.fshr.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_ror_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_ror_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_ror_epi64
  // CHECK: @llvm.fshr.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_ror_epi64(__U, __A, 5); 
}


__m512i test_mm512_rorv_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_rorv_epi32
  // CHECK: @llvm.fshr.v16i32
  return _mm512_rorv_epi32(__A, __B); 
}

__m512i test_mm512_mask_rorv_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_rorv_epi32
  // CHECK: @llvm.fshr.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_rorv_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_rorv_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_rorv_epi32
  // CHECK: @llvm.fshr.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_rorv_epi32(__U, __A, __B); 
}

__m512i test_mm512_rorv_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_rorv_epi64
  // CHECK: @llvm.fshr.v8i64
  return _mm512_rorv_epi64(__A, __B); 
}

__m512i test_mm512_mask_rorv_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_rorv_epi64
  // CHECK: @llvm.fshr.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_rorv_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_rorv_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_rorv_epi64
  // CHECK: @llvm.fshr.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_rorv_epi64(__U, __A, __B); 
}

__m512i test_mm512_slli_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_slli_epi32
  // CHECK: @llvm.x86.avx512.pslli.d.512
  return _mm512_slli_epi32(__A, 5); 
}

__m512i test_mm512_slli_epi32_2(__m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_slli_epi32_2
  // CHECK: @llvm.x86.avx512.pslli.d.512
  return _mm512_slli_epi32(__A, __B); 
}

__m512i test_mm512_mask_slli_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_slli_epi32
  // CHECK: @llvm.x86.avx512.pslli.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_slli_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_mask_slli_epi32_2(__m512i __W, __mmask16 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_mask_slli_epi32_2
  // CHECK: @llvm.x86.avx512.pslli.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_slli_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_slli_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_slli_epi32
  // CHECK: @llvm.x86.avx512.pslli.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_slli_epi32(__U, __A, 5); 
}

__m512i test_mm512_maskz_slli_epi32_2(__mmask16 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_maskz_slli_epi32_2
  // CHECK: @llvm.x86.avx512.pslli.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_slli_epi32(__U, __A, __B); 
}

__m512i test_mm512_slli_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_slli_epi64
  // CHECK: @llvm.x86.avx512.pslli.q.512
  return _mm512_slli_epi64(__A, 5); 
}

__m512i test_mm512_slli_epi64_2(__m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_slli_epi64_2
  // CHECK: @llvm.x86.avx512.pslli.q.512
  return _mm512_slli_epi64(__A, __B); 
}

__m512i test_mm512_mask_slli_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_slli_epi64
  // CHECK: @llvm.x86.avx512.pslli.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_slli_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_mask_slli_epi64_2(__m512i __W, __mmask8 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_mask_slli_epi64_2
  // CHECK: @llvm.x86.avx512.pslli.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_slli_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_slli_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_slli_epi64
  // CHECK: @llvm.x86.avx512.pslli.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_slli_epi64(__U, __A, 5); 
}

__m512i test_mm512_maskz_slli_epi64_2(__mmask8 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_maskz_slli_epi64_2
  // CHECK: @llvm.x86.avx512.pslli.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_slli_epi64(__U, __A, __B); 
}

__m512i test_mm512_srli_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_srli_epi32
  // CHECK: @llvm.x86.avx512.psrli.d.512
  return _mm512_srli_epi32(__A, 5); 
}

__m512i test_mm512_srli_epi32_2(__m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_srli_epi32_2
  // CHECK: @llvm.x86.avx512.psrli.d.512
  return _mm512_srli_epi32(__A, __B); 
}

__m512i test_mm512_mask_srli_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_srli_epi32
  // CHECK: @llvm.x86.avx512.psrli.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_srli_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_mask_srli_epi32_2(__m512i __W, __mmask16 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_mask_srli_epi32_2
  // CHECK: @llvm.x86.avx512.psrli.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_srli_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srli_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_srli_epi32
  // CHECK: @llvm.x86.avx512.psrli.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_srli_epi32(__U, __A, 5); 
}

__m512i test_mm512_maskz_srli_epi32_2(__mmask16 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_maskz_srli_epi32_2
  // CHECK: @llvm.x86.avx512.psrli.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_srli_epi32(__U, __A, __B); 
}

__m512i test_mm512_srli_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_srli_epi64
  // CHECK: @llvm.x86.avx512.psrli.q.512
  return _mm512_srli_epi64(__A, 5); 
}

__m512i test_mm512_srli_epi64_2(__m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_srli_epi64_2
  // CHECK: @llvm.x86.avx512.psrli.q.512
  return _mm512_srli_epi64(__A, __B); 
}

__m512i test_mm512_mask_srli_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_srli_epi64
  // CHECK: @llvm.x86.avx512.psrli.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_srli_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_mask_srli_epi64_2(__m512i __W, __mmask8 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_mask_srli_epi64_2
  // CHECK: @llvm.x86.avx512.psrli.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_srli_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srli_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_srli_epi64
  // CHECK: @llvm.x86.avx512.psrli.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_srli_epi64(__U, __A, 5); 
}

__m512i test_mm512_maskz_srli_epi64_2(__mmask8 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_maskz_srli_epi64_2
  // CHECK: @llvm.x86.avx512.psrli.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_srli_epi64(__U, __A, __B); 
}

__m512i test_mm512_mask_load_epi32(__m512i __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_mask_load_epi32
  // CHECK: @llvm.masked.load.v16i32.p0(ptr %{{.*}}, i32 64, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_mask_load_epi32(__W, __U, __P); 
}

__m512i test_mm512_maskz_load_epi32(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_maskz_load_epi32
  // CHECK: @llvm.masked.load.v16i32.p0(ptr %{{.*}}, i32 64, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_maskz_load_epi32(__U, __P); 
}

__m512i test_mm512_mask_mov_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_mov_epi32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_mov_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_mov_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_mov_epi32
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_mov_epi32(__U, __A); 
}

__m512i test_mm512_mask_mov_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_mov_epi64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_mov_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_mov_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_mov_epi64
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_mov_epi64(__U, __A); 
}

__m512i test_mm512_mask_load_epi64(__m512i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_mask_load_epi64
  // CHECK: @llvm.masked.load.v8i64.p0(ptr %{{.*}}, i32 64, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_mask_load_epi64(__W, __U, __P); 
}

__m512i test_mm512_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_maskz_load_epi64
  // CHECK: @llvm.masked.load.v8i64.p0(ptr %{{.*}}, i32 64, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_maskz_load_epi64(__U, __P); 
}

void test_mm512_mask_store_epi32(void *__P, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_store_epi32
  // CHECK: @llvm.masked.store.v16i32.p0(<16 x i32> %{{.*}}, ptr %{{.*}}, i32 64, <16 x i1> %{{.*}})
  return _mm512_mask_store_epi32(__P, __U, __A); 
}

void test_mm512_mask_store_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_store_epi64
  // CHECK: @llvm.masked.store.v8i64.p0(<8 x i64> %{{.*}}, ptr %{{.*}}, i32 64, <8 x i1> %{{.*}})
  return _mm512_mask_store_epi64(__P, __U, __A); 
}

__m512d test_mm512_movedup_pd(__m512d __A) {
  // CHECK-LABEL: test_mm512_movedup_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  return _mm512_movedup_pd(__A);
}

__m512d test_mm512_mask_movedup_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_movedup_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_movedup_pd(__W, __U, __A);
}

__m512d test_mm512_maskz_movedup_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_movedup_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_movedup_pd(__U, __A);
}

int test_mm_comi_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_comi_round_sd
  // CHECK: @llvm.x86.avx512.vcomi.sd
  return _mm_comi_round_sd(__A, __B, 5, _MM_FROUND_NO_EXC); 
}

int test_mm_comi_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_comi_round_ss
  // CHECK: @llvm.x86.avx512.vcomi.ss
  return _mm_comi_round_ss(__A, __B, 5, _MM_FROUND_NO_EXC); 
}

__m512d test_mm512_fixupimm_round_pd(__m512d __A, __m512d __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_fixupimm_round_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.512
  return _mm512_fixupimm_round_pd(__A, __B, __C, 5, 8); 
}

__m512d test_mm512_mask_fixupimm_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_mask_fixupimm_round_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.512
  return _mm512_mask_fixupimm_round_pd(__A, __U, __B, __C, 5, 8); 
}

__m512d test_mm512_fixupimm_pd(__m512d __A, __m512d __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.512
  return _mm512_fixupimm_pd(__A, __B, __C, 5); 
}

__m512d test_mm512_mask_fixupimm_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_mask_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.512
  return _mm512_mask_fixupimm_pd(__A, __U, __B, __C, 5); 
}

__m512d test_mm512_maskz_fixupimm_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_maskz_fixupimm_round_pd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.pd.512
  return _mm512_maskz_fixupimm_round_pd(__U, __A, __B, __C, 5, 8); 
}

__m512d test_mm512_maskz_fixupimm_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_maskz_fixupimm_pd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.pd.512
  return _mm512_maskz_fixupimm_pd(__U, __A, __B, __C, 5); 
}

__m512 test_mm512_fixupimm_round_ps(__m512 __A, __m512 __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_fixupimm_round_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.512
  return _mm512_fixupimm_round_ps(__A, __B, __C, 5, 8); 
}

__m512 test_mm512_mask_fixupimm_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_mask_fixupimm_round_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.512
  return _mm512_mask_fixupimm_round_ps(__A, __U, __B, __C, 5, 8); 
}

__m512 test_mm512_fixupimm_ps(__m512 __A, __m512 __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.512
  return _mm512_fixupimm_ps(__A, __B, __C, 5); 
}

__m512 test_mm512_mask_fixupimm_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_mask_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.512
  return _mm512_mask_fixupimm_ps(__A, __U, __B, __C, 5); 
}

__m512 test_mm512_maskz_fixupimm_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_maskz_fixupimm_round_ps
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.ps.512
  return _mm512_maskz_fixupimm_round_ps(__U, __A, __B, __C, 5, 8); 
}

__m512 test_mm512_maskz_fixupimm_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_maskz_fixupimm_ps
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.ps.512
  return _mm512_maskz_fixupimm_ps(__U, __A, __B, __C, 5); 
}

__m128d test_mm_fixupimm_round_sd(__m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: test_mm_fixupimm_round_sd
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_fixupimm_round_sd(__A, __B, __C, 5, 8); 
}

__m128d test_mm_mask_fixupimm_round_sd(__m128d __A, __mmask8 __U, __m128d __B, __m128i __C) {
  // CHECK-LABEL: test_mm_mask_fixupimm_round_sd
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_mask_fixupimm_round_sd(__A, __U, __B, __C, 5, 8); 
}

__m128d test_mm_fixupimm_sd(__m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: test_mm_fixupimm_sd
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_fixupimm_sd(__A, __B, __C, 5); 
}

__m128d test_mm_mask_fixupimm_sd(__m128d __A, __mmask8 __U, __m128d __B, __m128i __C) {
  // CHECK-LABEL: test_mm_mask_fixupimm_sd
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_mask_fixupimm_sd(__A, __U, __B, __C, 5); 
}

__m128d test_mm_maskz_fixupimm_round_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: test_mm_maskz_fixupimm_round_sd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm
  return _mm_maskz_fixupimm_round_sd(__U, __A, __B, __C, 5, 8); 
}

__m128d test_mm_maskz_fixupimm_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: test_mm_maskz_fixupimm_sd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm
  return _mm_maskz_fixupimm_sd(__U, __A, __B, __C, 5); 
}

__m128 test_mm_fixupimm_round_ss(__m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: test_mm_fixupimm_round_ss
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_fixupimm_round_ss(__A, __B, __C, 5, 8); 
}

__m128 test_mm_mask_fixupimm_round_ss(__m128 __A, __mmask8 __U, __m128 __B, __m128i __C) {
  // CHECK-LABEL: test_mm_mask_fixupimm_round_ss
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_mask_fixupimm_round_ss(__A, __U, __B, __C, 5, 8); 
}

__m128 test_mm_fixupimm_ss(__m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: test_mm_fixupimm_ss
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_fixupimm_ss(__A, __B, __C, 5); 
}

__m128 test_mm_mask_fixupimm_ss(__m128 __A, __mmask8 __U, __m128 __B, __m128i __C) {
  // CHECK-LABEL: test_mm_mask_fixupimm_ss
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_mask_fixupimm_ss(__A, __U, __B, __C, 5); 
}

__m128 test_mm_maskz_fixupimm_round_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: test_mm_maskz_fixupimm_round_ss
  // CHECK: @llvm.x86.avx512.maskz.fixupimm
  return _mm_maskz_fixupimm_round_ss(__U, __A, __B, __C, 5, 8); 
}

__m128 test_mm_maskz_fixupimm_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: test_mm_maskz_fixupimm_ss
  // CHECK: @llvm.x86.avx512.maskz.fixupimm
  return _mm_maskz_fixupimm_ss(__U, __A, __B, __C, 5); 
}

__m128d test_mm_getexp_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_getexp_round_sd
  // CHECK: @llvm.x86.avx512.mask.getexp.sd
  return _mm_getexp_round_sd(__A, __B, 8); 
}

__m128d test_mm_getexp_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_getexp_sd
  // CHECK: @llvm.x86.avx512.mask.getexp.sd
  return _mm_getexp_sd(__A, __B); 
}

__m128 test_mm_getexp_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_getexp_round_ss
  // CHECK: @llvm.x86.avx512.mask.getexp.ss
  return _mm_getexp_round_ss(__A, __B, 8); 
}

__m128 test_mm_getexp_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_getexp_ss
  // CHECK: @llvm.x86.avx512.mask.getexp.ss
  return _mm_getexp_ss(__A, __B); 
}

__m128d test_mm_getmant_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_getmant_round_sd
  // CHECK: @llvm.x86.avx512.mask.getmant.sd
  return _mm_getmant_round_sd(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src, 8); 
}

__m128d test_mm_getmant_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_getmant_sd
  // CHECK: @llvm.x86.avx512.mask.getmant.sd
  return _mm_getmant_sd(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src); 
}

__m128 test_mm_getmant_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_getmant_round_ss
  // CHECK: @llvm.x86.avx512.mask.getmant.ss
  return _mm_getmant_round_ss(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src, 8); 
}

__m128 test_mm_getmant_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_getmant_ss
  // CHECK: @llvm.x86.avx512.mask.getmant.ss
  return _mm_getmant_ss(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src); 
}

__mmask16 test_mm512_kmov(__mmask16 __A) {
  // CHECK-LABEL: test_mm512_kmov
  // CHECK: load i16, ptr %__A.addr.i, align 2{{$}}
  return _mm512_kmov(__A); 
}

__m512d test_mm512_mask_unpackhi_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_unpackhi_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_unpackhi_pd(__W, __U, __A, __B); 
}
#if __x86_64__
long long test_mm_cvt_roundsd_si64(__m128d __A) {
  // CHECK-LABEL: test_mm_cvt_roundsd_si64
  // CHECK: @llvm.x86.avx512.vcvtsd2si64
  return _mm_cvt_roundsd_si64(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
#endif
__m512i test_mm512_mask2_permutex2var_epi32(__m512i __A, __m512i __I, __mmask16 __U, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask2_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask2_permutex2var_epi32(__A, __I, __U, __B); 
}
__m512i test_mm512_unpackhi_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_unpackhi_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  return _mm512_unpackhi_epi32(__A, __B); 
}

__m512d test_mm512_maskz_unpackhi_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_unpackhi_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_unpackhi_pd(__U, __A, __B); 
}
#if __x86_64__
long long test_mm_cvt_roundsd_i64(__m128d __A) {
  // CHECK-LABEL: test_mm_cvt_roundsd_i64
  // CHECK: @llvm.x86.avx512.vcvtsd2si64
  return _mm_cvt_roundsd_i64(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
#endif
__m512d test_mm512_mask2_permutex2var_pd(__m512d __A, __m512i __I, __mmask8 __U, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask2_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask2_permutex2var_pd(__A, __I, __U, __B); 
}
__m512i test_mm512_mask_unpackhi_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_unpackhi_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_unpackhi_epi32(__W, __U, __A, __B); 
}

__m512 test_mm512_mask_unpackhi_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_unpackhi_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_unpackhi_ps(__W, __U, __A, __B); 
}

__m512 test_mm512_maskz_unpackhi_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_unpackhi_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_unpackhi_ps(__U, __A, __B); 
}

__m512d test_mm512_mask_unpacklo_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_unpacklo_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_unpacklo_pd(__W, __U, __A, __B); 
}

__m512d test_mm512_maskz_unpacklo_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_unpacklo_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_unpacklo_pd(__U, __A, __B); 
}

__m512 test_mm512_mask_unpacklo_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_unpacklo_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_unpacklo_ps(__W, __U, __A, __B); 
}

__m512 test_mm512_maskz_unpacklo_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_unpacklo_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_unpacklo_ps(__U, __A, __B); 
}
int test_mm_cvt_roundsd_si32(__m128d __A) {
  // CHECK-LABEL: test_mm_cvt_roundsd_si32
  // CHECK: @llvm.x86.avx512.vcvtsd2si32
  return _mm_cvt_roundsd_si32(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

int test_mm_cvt_roundsd_i32(__m128d __A) {
  // CHECK-LABEL: test_mm_cvt_roundsd_i32
  // CHECK: @llvm.x86.avx512.vcvtsd2si32
  return _mm_cvt_roundsd_i32(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

unsigned test_mm_cvt_roundsd_u32(__m128d __A) {
  // CHECK-LABEL: test_mm_cvt_roundsd_u32
  // CHECK: @llvm.x86.avx512.vcvtsd2usi32
  return _mm_cvt_roundsd_u32(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

unsigned test_mm_cvtsd_u32(__m128d __A) {
  // CHECK-LABEL: test_mm_cvtsd_u32
  // CHECK: @llvm.x86.avx512.vcvtsd2usi32
  return _mm_cvtsd_u32(__A); 
}

int test_mm512_cvtsi512_si32(__m512i a) {
  // CHECK-LABEL: test_mm512_cvtsi512_si32
  // CHECK: %{{.*}} = extractelement <16 x i32> %{{.*}}, i32 0
  return _mm512_cvtsi512_si32(a);
}

#ifdef __x86_64__
unsigned long long test_mm_cvt_roundsd_u64(__m128d __A) {
  // CHECK-LABEL: test_mm_cvt_roundsd_u64
  // CHECK: @llvm.x86.avx512.vcvtsd2usi64
  return _mm_cvt_roundsd_u64(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

unsigned long long test_mm_cvtsd_u64(__m128d __A) {
  // CHECK-LABEL: test_mm_cvtsd_u64
  // CHECK: @llvm.x86.avx512.vcvtsd2usi64
  return _mm_cvtsd_u64(__A); 
}
#endif

int test_mm_cvt_roundss_si32(__m128 __A) {
  // CHECK-LABEL: test_mm_cvt_roundss_si32
  // CHECK: @llvm.x86.avx512.vcvtss2si32
  return _mm_cvt_roundss_si32(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

int test_mm_cvt_roundss_i32(__m128 __A) {
  // CHECK-LABEL: test_mm_cvt_roundss_i32
  // CHECK: @llvm.x86.avx512.vcvtss2si32
  return _mm_cvt_roundss_i32(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

#ifdef __x86_64__
long long test_mm_cvt_roundss_si64(__m128 __A) {
  // CHECK-LABEL: test_mm_cvt_roundss_si64
  // CHECK: @llvm.x86.avx512.vcvtss2si64
  return _mm_cvt_roundss_si64(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

long long test_mm_cvt_roundss_i64(__m128 __A) {
  // CHECK-LABEL: test_mm_cvt_roundss_i64
  // CHECK: @llvm.x86.avx512.vcvtss2si64
  return _mm_cvt_roundss_i64(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
#endif

unsigned test_mm_cvt_roundss_u32(__m128 __A) {
  // CHECK-LABEL: test_mm_cvt_roundss_u32
  // CHECK: @llvm.x86.avx512.vcvtss2usi32
  return _mm_cvt_roundss_u32(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

unsigned test_mm_cvtss_u32(__m128 __A) {
  // CHECK-LABEL: test_mm_cvtss_u32
  // CHECK: @llvm.x86.avx512.vcvtss2usi32
  return _mm_cvtss_u32(__A); 
}

#ifdef __x86_64__
unsigned long long test_mm_cvt_roundss_u64(__m128 __A) {
  // CHECK-LABEL: test_mm_cvt_roundss_u64
  // CHECK: @llvm.x86.avx512.vcvtss2usi64
  return _mm_cvt_roundss_u64(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

unsigned long long test_mm_cvtss_u64(__m128 __A) {
  // CHECK-LABEL: test_mm_cvtss_u64
  // CHECK: @llvm.x86.avx512.vcvtss2usi64
  return _mm_cvtss_u64(__A); 
}
#endif

int test_mm_cvtt_roundsd_i32(__m128d __A) {
  // CHECK-LABEL: test_mm_cvtt_roundsd_i32
  // CHECK: @llvm.x86.avx512.cvttsd2si
  return _mm_cvtt_roundsd_i32(__A, _MM_FROUND_NO_EXC);
}

int test_mm_cvtt_roundsd_si32(__m128d __A) {
  // CHECK-LABEL: test_mm_cvtt_roundsd_si32
  // CHECK: @llvm.x86.avx512.cvttsd2si
  return _mm_cvtt_roundsd_si32(__A, _MM_FROUND_NO_EXC);
}

int test_mm_cvttsd_i32(__m128d __A) {
  // CHECK-LABEL: test_mm_cvttsd_i32
  // CHECK: @llvm.x86.avx512.cvttsd2si
  return _mm_cvttsd_i32(__A); 
}

#ifdef __x86_64__
long long test_mm_cvtt_roundsd_si64(__m128d __A) {
  // CHECK-LABEL: test_mm_cvtt_roundsd_si64
  // CHECK: @llvm.x86.avx512.cvttsd2si64
  return _mm_cvtt_roundsd_si64(__A, _MM_FROUND_NO_EXC);
}

long long test_mm_cvtt_roundsd_i64(__m128d __A) {
  // CHECK-LABEL: test_mm_cvtt_roundsd_i64
  // CHECK: @llvm.x86.avx512.cvttsd2si64
  return _mm_cvtt_roundsd_i64(__A, _MM_FROUND_NO_EXC);
}

long long test_mm_cvttsd_i64(__m128d __A) {
  // CHECK-LABEL: test_mm_cvttsd_i64
  // CHECK: @llvm.x86.avx512.cvttsd2si64
  return _mm_cvttsd_i64(__A); 
}
#endif

unsigned test_mm_cvtt_roundsd_u32(__m128d __A) {
  // CHECK-LABEL: test_mm_cvtt_roundsd_u32
  // CHECK: @llvm.x86.avx512.cvttsd2usi
  return _mm_cvtt_roundsd_u32(__A, _MM_FROUND_NO_EXC);
}

unsigned test_mm_cvttsd_u32(__m128d __A) {
  // CHECK-LABEL: test_mm_cvttsd_u32
  // CHECK: @llvm.x86.avx512.cvttsd2usi
  return _mm_cvttsd_u32(__A); 
}

#ifdef __x86_64__
unsigned long long test_mm_cvtt_roundsd_u64(__m128d __A) {
  // CHECK-LABEL: test_mm_cvtt_roundsd_u64
  // CHECK: @llvm.x86.avx512.cvttsd2usi64
  return _mm_cvtt_roundsd_u64(__A, _MM_FROUND_NO_EXC);
}

unsigned long long test_mm_cvttsd_u64(__m128d __A) {
  // CHECK-LABEL: test_mm_cvttsd_u64
  // CHECK: @llvm.x86.avx512.cvttsd2usi64
  return _mm_cvttsd_u64(__A); 
}
#endif

int test_mm_cvtt_roundss_i32(__m128 __A) {
  // CHECK-LABEL: test_mm_cvtt_roundss_i32
  // CHECK: @llvm.x86.avx512.cvttss2si
  return _mm_cvtt_roundss_i32(__A, _MM_FROUND_NO_EXC);
}

int test_mm_cvtt_roundss_si32(__m128 __A) {
  // CHECK-LABEL: test_mm_cvtt_roundss_si32
  // CHECK: @llvm.x86.avx512.cvttss2si
  return _mm_cvtt_roundss_si32(__A, _MM_FROUND_NO_EXC);
}

int test_mm_cvttss_i32(__m128 __A) {
  // CHECK-LABEL: test_mm_cvttss_i32
  // CHECK: @llvm.x86.avx512.cvttss2si
  return _mm_cvttss_i32(__A); 
}

#ifdef __x86_64__
float test_mm_cvtt_roundss_i64(__m128 __A) {
  // CHECK-LABEL: test_mm_cvtt_roundss_i64
  // CHECK: @llvm.x86.avx512.cvttss2si64
  return _mm_cvtt_roundss_i64(__A, _MM_FROUND_NO_EXC);
}

long long test_mm_cvtt_roundss_si64(__m128 __A) {
  // CHECK-LABEL: test_mm_cvtt_roundss_si64
  // CHECK: @llvm.x86.avx512.cvttss2si64
  return _mm_cvtt_roundss_si64(__A, _MM_FROUND_NO_EXC);
}

long long test_mm_cvttss_i64(__m128 __A) {
  // CHECK-LABEL: test_mm_cvttss_i64
  // CHECK: @llvm.x86.avx512.cvttss2si64
  return _mm_cvttss_i64(__A); 
}
#endif

unsigned test_mm_cvtt_roundss_u32(__m128 __A) {
  // CHECK-LABEL: test_mm_cvtt_roundss_u32
  // CHECK: @llvm.x86.avx512.cvttss2usi
  return _mm_cvtt_roundss_u32(__A, _MM_FROUND_NO_EXC);
}

unsigned test_mm_cvttss_u32(__m128 __A) {
  // CHECK-LABEL: test_mm_cvttss_u32
  // CHECK: @llvm.x86.avx512.cvttss2usi
  return _mm_cvttss_u32(__A); 
}

#ifdef __x86_64__
unsigned long long test_mm_cvtt_roundss_u64(__m128 __A) {
  // CHECK-LABEL: test_mm_cvtt_roundss_u64
  // CHECK: @llvm.x86.avx512.cvttss2usi64
  return _mm_cvtt_roundss_u64(__A, _MM_FROUND_NO_EXC);
}

unsigned long long test_mm_cvttss_u64(__m128 __A) {
  // CHECK-LABEL: test_mm_cvttss_u64
  // CHECK: @llvm.x86.avx512.cvttss2usi64
  return _mm_cvttss_u64(__A); 
}
#endif

__m512i test_mm512_cvtt_roundps_epu32(__m512 __A) 
{
    // CHECK-LABEL: test_mm512_cvtt_roundps_epu32
    // CHECK: @llvm.x86.avx512.mask.cvttps2udq.512
    return _mm512_cvtt_roundps_epu32(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundps_epu32(__m512i __W, __mmask16 __U, __m512 __A)
{
    // CHECK-LABEL: test_mm512_mask_cvtt_roundps_epu32
    // CHECK: @llvm.x86.avx512.mask.cvttps2udq.512
    return _mm512_mask_cvtt_roundps_epu32(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundps_epu32( __mmask16 __U, __m512 __A)
{
    // CHECK-LABEL: test_mm512_maskz_cvtt_roundps_epu32
    // CHECK: @llvm.x86.avx512.mask.cvttps2udq.512

    return _mm512_maskz_cvtt_roundps_epu32(__U, __A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_cvt_roundps_ph(__m512  __A)
{
    // CHECK-LABEL: test_mm512_cvt_roundps_ph
    // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.512
    return _mm512_cvt_roundps_ph(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvt_roundps_ph(__m256i __W , __mmask16 __U, __m512  __A)
{
    // CHECK-LABEL: test_mm512_mask_cvt_roundps_ph
    // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.512
    return _mm512_mask_cvt_roundps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvt_roundps_ph(__mmask16 __U, __m512  __A)
{
    // CHECK-LABEL: test_mm512_maskz_cvt_roundps_ph
    // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.512
    return _mm512_maskz_cvt_roundps_ph(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_cvt_roundph_ps(__m256i __A)
{
    // CHECK-LABEL: test_mm512_cvt_roundph_ps
    // CHECK: @llvm.x86.avx512.mask.vcvtph2ps.512(
    return _mm512_cvt_roundph_ps(__A, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_cvt_roundph_ps(__m512 __W, __mmask16 __U, __m256i __A)
{
    // CHECK-LABEL: test_mm512_mask_cvt_roundph_ps
    // CHECK: @llvm.x86.avx512.mask.vcvtph2ps.512(
    return _mm512_mask_cvt_roundph_ps(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_cvt_roundph_ps(__mmask16 __U, __m256i __A)
{
    // CHECK-LABEL: test_mm512_maskz_cvt_roundph_ps
    // CHECK: @llvm.x86.avx512.mask.vcvtph2ps.512(
    return _mm512_maskz_cvt_roundph_ps(__U, __A, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_cvt_roundepi32_ps( __m512i __A)
{
  // CHECK-LABEL: test_mm512_cvt_roundepi32_ps
  // CHECK: @llvm.x86.avx512.sitofp.round.v16f32.v16i32
  return _mm512_cvt_roundepi32_ps(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_cvt_roundepi32_ps(__m512 __W, __mmask16 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_mask_cvt_roundepi32_ps
  // CHECK: @llvm.x86.avx512.sitofp.round.v16f32.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_cvt_roundepi32_ps(__W,__U,__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_cvt_roundepi32_ps(__mmask16 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepi32_ps
  // CHECK: @llvm.x86.avx512.sitofp.round.v16f32.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_cvt_roundepi32_ps(__U,__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_cvt_roundepu32_ps(__m512i __A)
{
  // CHECK-LABEL: test_mm512_cvt_roundepu32_ps
  // CHECK: @llvm.x86.avx512.uitofp.round.v16f32.v16i32
  return _mm512_cvt_roundepu32_ps(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_cvt_roundepu32_ps(__m512 __W, __mmask16 __U,__m512i __A)
{
  // CHECK-LABEL: test_mm512_mask_cvt_roundepu32_ps
  // CHECK: @llvm.x86.avx512.uitofp.round.v16f32.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_cvt_roundepu32_ps(__W,__U,__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_cvt_roundepu32_ps(__mmask16 __U,__m512i __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepu32_ps
  // CHECK: @llvm.x86.avx512.uitofp.round.v16f32.v16i32
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_cvt_roundepu32_ps(__U,__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm512_cvt_roundpd_ps(__m512d A)
{
  // CHECK-LABEL: test_mm512_cvt_roundpd_ps
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.512
  return _mm512_cvt_roundpd_ps(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm512_mask_cvt_roundpd_ps(__m256 W, __mmask8 U,__m512d A)
{
  // CHECK-LABEL: test_mm512_mask_cvt_roundpd_ps
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.512
  return _mm512_mask_cvt_roundpd_ps(W,U,A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm512_maskz_cvt_roundpd_ps(__mmask8 U, __m512d A)
{
  // CHECK-LABEL: test_mm512_maskz_cvt_roundpd_ps
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.512
  return _mm512_maskz_cvt_roundpd_ps(U,A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_cvtt_roundpd_epi32(__m512d A)
{
  // CHECK-LABEL: test_mm512_cvtt_roundpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.512
  return _mm512_cvtt_roundpd_epi32(A,_MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvtt_roundpd_epi32(__m256i W, __mmask8 U, __m512d A)
{
  // CHECK-LABEL: test_mm512_mask_cvtt_roundpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.512
  return _mm512_mask_cvtt_roundpd_epi32(W,U,A,_MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvtt_roundpd_epi32(__mmask8 U, __m512d A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.512
  return _mm512_maskz_cvtt_roundpd_epi32(U,A,_MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvtt_roundps_epi32(__m512 A)
{
  // CHECK-LABEL: test_mm512_cvtt_roundps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttps2dq.512
  return _mm512_cvtt_roundps_epi32(A,_MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundps_epi32(__m512i W,__mmask16 U, __m512 A)
{
  // CHECK-LABEL: test_mm512_mask_cvtt_roundps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttps2dq.512
  return _mm512_mask_cvtt_roundps_epi32(W,U,A,_MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundps_epi32(__mmask16 U, __m512 A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttps2dq.512
  return _mm512_maskz_cvtt_roundps_epi32(U,A,_MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvt_roundps_epi32(__m512 __A)
{
  // CHECK-LABEL: test_mm512_cvt_roundps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.512
  return _mm512_cvt_roundps_epi32(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvt_roundps_epi32(__m512i __W,__mmask16 __U,__m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_cvt_roundps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.512
  return _mm512_mask_cvt_roundps_epi32(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvt_roundps_epi32(__mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvt_roundps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.512
  return _mm512_maskz_cvt_roundps_epi32(__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_cvt_roundpd_epi32(__m512d A)
{
  // CHECK-LABEL: test_mm512_cvt_roundpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.512
  return _mm512_cvt_roundpd_epi32(A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvt_roundpd_epi32(__m256i W,__mmask8 U,__m512d A)
{
  // CHECK-LABEL: test_mm512_mask_cvt_roundpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.512
  return _mm512_mask_cvt_roundpd_epi32(W,U,A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvt_roundpd_epi32(__mmask8 U, __m512d A)
{
  // CHECK-LABEL: test_mm512_maskz_cvt_roundpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.512
  return _mm512_maskz_cvt_roundpd_epi32(U,A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvt_roundps_epu32(__m512 __A)
{
  // CHECK-LABEL: test_mm512_cvt_roundps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.512
  return _mm512_cvt_roundps_epu32(__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvt_roundps_epu32(__m512i __W,__mmask16 __U,__m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_cvt_roundps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.512
  return _mm512_mask_cvt_roundps_epu32(__W,__U,__A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvt_roundps_epu32(__mmask16 __U,__m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvt_roundps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.512
  return _mm512_maskz_cvt_roundps_epu32(__U,__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_cvt_roundpd_epu32(__m512d A)
{
  // CHECK-LABEL: test_mm512_cvt_roundpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.512
  return _mm512_cvt_roundpd_epu32(A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvt_roundpd_epu32(__m256i W, __mmask8 U, __m512d A)
{
  // CHECK-LABEL: test_mm512_mask_cvt_roundpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.512
  return _mm512_mask_cvt_roundpd_epu32(W,U,A,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvt_roundpd_epu32(__mmask8 U, __m512d A) 
{
  // CHECK-LABEL: test_mm512_maskz_cvt_roundpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.512
  return _mm512_maskz_cvt_roundpd_epu32(U, A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask2_permutex2var_ps(__m512 __A, __m512i __I, __mmask16 __U, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask2_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask2_permutex2var_ps(__A, __I, __U, __B); 
}

__m512i test_mm512_mask2_permutex2var_epi64(__m512i __A, __m512i __I, __mmask8 __U, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask2_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask2_permutex2var_epi64(__A, __I, __U, __B); 
}

__m512d test_mm512_permute_pd(__m512d __X) {
  // CHECK-LABEL: test_mm512_permute_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  return _mm512_permute_pd(__X, 2);
}

__m512d test_mm512_mask_permute_pd(__m512d __W, __mmask8 __U, __m512d __X) {
  // CHECK-LABEL: test_mm512_mask_permute_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_permute_pd(__W, __U, __X, 2);
}

__m512d test_mm512_maskz_permute_pd(__mmask8 __U, __m512d __X) {
  // CHECK-LABEL: test_mm512_maskz_permute_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_permute_pd(__U, __X, 2);
}

__m512 test_mm512_permute_ps(__m512 __X) {
  // CHECK-LABEL: test_mm512_permute_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <16 x i32> <i32 2, i32 0, i32 0, i32 0, i32 6, i32 4, i32 4, i32 4, i32 10, i32 8, i32 8, i32 8, i32 14, i32 12, i32 12, i32 12>
  return _mm512_permute_ps(__X, 2);
}

__m512 test_mm512_mask_permute_ps(__m512 __W, __mmask16 __U, __m512 __X) {
  // CHECK-LABEL: test_mm512_mask_permute_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <16 x i32> <i32 2, i32 0, i32 0, i32 0, i32 6, i32 4, i32 4, i32 4, i32 10, i32 8, i32 8, i32 8, i32 14, i32 12, i32 12, i32 12>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_permute_ps(__W, __U, __X, 2);
}

__m512 test_mm512_maskz_permute_ps(__mmask16 __U, __m512 __X) {
  // CHECK-LABEL: test_mm512_maskz_permute_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <16 x i32> <i32 2, i32 0, i32 0, i32 0, i32 6, i32 4, i32 4, i32 4, i32 10, i32 8, i32 8, i32 8, i32 14, i32 12, i32 12, i32 12>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_permute_ps(__U, __X, 2);
}

__m512d test_mm512_permutevar_pd(__m512d __A, __m512i __C) {
  // CHECK-LABEL: test_mm512_permutevar_pd
  // CHECK: @llvm.x86.avx512.vpermilvar.pd.512
  return _mm512_permutevar_pd(__A, __C); 
}

__m512d test_mm512_mask_permutevar_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512i __C) {
  // CHECK-LABEL: test_mm512_mask_permutevar_pd
  // CHECK: @llvm.x86.avx512.vpermilvar.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_permutevar_pd(__W, __U, __A, __C); 
}

__m512d test_mm512_maskz_permutevar_pd(__mmask8 __U, __m512d __A, __m512i __C) {
  // CHECK-LABEL: test_mm512_maskz_permutevar_pd
  // CHECK: @llvm.x86.avx512.vpermilvar.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_permutevar_pd(__U, __A, __C); 
}

__m512 test_mm512_permutevar_ps(__m512 __A, __m512i __C) {
  // CHECK-LABEL: test_mm512_permutevar_ps
  // CHECK: @llvm.x86.avx512.vpermilvar.ps.512
  return _mm512_permutevar_ps(__A, __C); 
}

__m512 test_mm512_mask_permutevar_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512i __C) {
  // CHECK-LABEL: test_mm512_mask_permutevar_ps
  // CHECK: @llvm.x86.avx512.vpermilvar.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_permutevar_ps(__W, __U, __A, __C); 
}

__m512 test_mm512_maskz_permutevar_ps(__mmask16 __U, __m512 __A, __m512i __C) {
  // CHECK-LABEL: test_mm512_maskz_permutevar_ps
  // CHECK: @llvm.x86.avx512.vpermilvar.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_permutevar_ps(__U, __A, __C); 
}

__m512i test_mm512_permutex2var_epi32(__m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: test_mm512_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.512
  return _mm512_permutex2var_epi32(__A, __I, __B); 
}

__m512i test_mm512_maskz_permutex2var_epi32(__mmask16 __U, __m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_permutex2var_epi32(__U, __A, __I, __B); 
}

__m512i test_mm512_mask_permutex2var_epi32 (__m512i __A, __mmask16 __U, __m512i __I, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_permutex2var_epi32 
  // CHECK: @llvm.x86.avx512.vpermi2var.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_permutex2var_epi32 (__A,__U,__I,__B);
}

__m512d test_mm512_permutex2var_pd (__m512d __A, __m512i __I, __m512d __B)
{
  // CHECK-LABEL: test_mm512_permutex2var_pd 
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.512
  return _mm512_permutex2var_pd (__A, __I,__B);
}

__m512d test_mm512_mask_permutex2var_pd (__m512d __A, __mmask8 __U, __m512i __I, __m512d __B)
{
  // CHECK-LABEL: test_mm512_mask_permutex2var_pd 
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_permutex2var_pd (__A,__U,__I,__B);
}

__m512d test_mm512_maskz_permutex2var_pd(__mmask8 __U, __m512d __A, __m512i __I, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_permutex2var_pd(__U, __A, __I, __B); 
}

__m512 test_mm512_permutex2var_ps (__m512 __A, __m512i __I, __m512 __B)
{
  // CHECK-LABEL: test_mm512_permutex2var_ps 
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.512
  return _mm512_permutex2var_ps (__A, __I, __B);
}

__m512 test_mm512_mask_permutex2var_ps (__m512 __A, __mmask16 __U, __m512i __I, __m512 __B)
{
  // CHECK-LABEL: test_mm512_mask_permutex2var_ps 
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_permutex2var_ps (__A,__U,__I,__B);
}

__m512 test_mm512_maskz_permutex2var_ps(__mmask16 __U, __m512 __A, __m512i __I, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_permutex2var_ps(__U, __A, __I, __B); 
}

__m512i test_mm512_permutex2var_epi64 (__m512i __A, __m512i __I, __m512i __B){
  // CHECK-LABEL: test_mm512_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.512
  return _mm512_permutex2var_epi64(__A, __I, __B);
}

__m512i test_mm512_mask_permutex2var_epi64 (__m512i __A, __mmask8 __U, __m512i __I, __m512i __B){
  // CHECK-LABEL: test_mm512_mask_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_permutex2var_epi64(__A, __U, __I, __B);
}

__m512i test_mm512_maskz_permutex2var_epi64(__mmask8 __U, __m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_permutex2var_epi64(__U, __A, __I, __B);
}
__mmask16 test_mm512_testn_epi32_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_testn_epi32_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  return _mm512_testn_epi32_mask(__A, __B); 
}

__mmask16 test_mm512_mask_testn_epi32_mask(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_testn_epi32_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_testn_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm512_testn_epi64_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_testn_epi64_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  return _mm512_testn_epi64_mask(__A, __B); 
}

__mmask8 test_mm512_mask_testn_epi64_mask(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_testn_epi64_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_testn_epi64_mask(__U, __A, __B); 
}

__mmask16 test_mm512_mask_test_epi32_mask (__mmask16 __U, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_test_epi32_mask 
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i32> %{{.*}}, %{{.*}}
  return _mm512_mask_test_epi32_mask (__U,__A,__B);
}

__mmask8 test_mm512_mask_test_epi64_mask (__mmask8 __U, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_test_epi64_mask 
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_test_epi64_mask (__U,__A,__B);
}

__m512i test_mm512_maskz_unpackhi_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_unpackhi_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_unpackhi_epi32(__U, __A, __B); 
}

__m512i test_mm512_unpackhi_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_unpackhi_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  return _mm512_unpackhi_epi64(__A, __B); 
}

__m512i test_mm512_mask_unpackhi_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_unpackhi_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_unpackhi_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpackhi_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_unpackhi_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_unpackhi_epi64(__U, __A, __B); 
}

__m512i test_mm512_unpacklo_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_unpacklo_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  return _mm512_unpacklo_epi32(__A, __B); 
}

__m512i test_mm512_mask_unpacklo_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_unpacklo_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_unpacklo_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpacklo_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_unpacklo_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_unpacklo_epi32(__U, __A, __B); 
}

__m512i test_mm512_unpacklo_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_unpacklo_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  return _mm512_unpacklo_epi64(__A, __B); 
}

__m512i test_mm512_mask_unpacklo_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_unpacklo_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_unpacklo_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpacklo_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_unpacklo_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_unpacklo_epi64(__U, __A, __B); 
}

__m128d test_mm_roundscale_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_roundscale_round_sd
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
  return _mm_roundscale_round_sd(__A, __B, 3, _MM_FROUND_NO_EXC); 
}

__m128d test_mm_roundscale_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_roundscale_sd
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
  return _mm_roundscale_sd(__A, __B, 3); 
}

__m128d test_mm_mask_roundscale_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
    return _mm_mask_roundscale_sd(__W,__U,__A,__B,3);
}

__m128d test_mm_mask_roundscale_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
    return _mm_mask_roundscale_round_sd(__W,__U,__A,__B,3,_MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_roundscale_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
    return _mm_maskz_roundscale_sd(__U,__A,__B,3);
}

__m128d test_mm_maskz_roundscale_round_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
    return _mm_maskz_roundscale_round_sd(__U,__A,__B,3,_MM_FROUND_NO_EXC );
}

__m128 test_mm_roundscale_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_roundscale_round_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
  return _mm_roundscale_round_ss(__A, __B, 3, _MM_FROUND_NO_EXC);
}

__m128 test_mm_roundscale_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_roundscale_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
  return _mm_roundscale_ss(__A, __B, 3); 
}

__m128 test_mm_mask_roundscale_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_roundscale_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
    return _mm_mask_roundscale_ss(__W,__U,__A,__B,3);
}

__m128 test_mm_maskz_roundscale_round_ss( __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_roundscale_round_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
    return _mm_maskz_roundscale_round_ss(__U,__A,__B,3,_MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_roundscale_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_roundscale_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
    return _mm_maskz_roundscale_ss(__U,__A,__B,3);
}

__m512d test_mm512_scalef_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_scalef_round_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_scalef_round_pd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_scalef_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_scalef_round_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_mask_scalef_round_pd(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_scalef_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_scalef_round_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_maskz_scalef_round_pd(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_scalef_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_scalef_pd(__A, __B); 
}

__m512d test_mm512_mask_scalef_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_mask_scalef_pd(__W, __U, __A, __B); 
}

__m512d test_mm512_maskz_scalef_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_maskz_scalef_pd(__U, __A, __B); 
}

__m512 test_mm512_scalef_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_scalef_round_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_scalef_round_ps(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_scalef_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_scalef_round_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_mask_scalef_round_ps(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_scalef_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_scalef_round_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_maskz_scalef_round_ps(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_scalef_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_scalef_ps(__A, __B); 
}

__m512 test_mm512_mask_scalef_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_mask_scalef_ps(__W, __U, __A, __B); 
}

__m512 test_mm512_maskz_scalef_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_maskz_scalef_ps(__U, __A, __B); 
}

__m128d test_mm_scalef_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_scalef_round_sd
  // CHECK: @llvm.x86.avx512.mask.scalef.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %2, i8 -1, i32 11)
  return _mm_scalef_round_sd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_scalef_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_scalef_sd
  // CHECK: @llvm.x86.avx512.mask.scalef
  return _mm_scalef_sd(__A, __B); 
}

__m128d test_mm_mask_scalef_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_scalef_sd
  // CHECK: @llvm.x86.avx512.mask.scalef.sd
  return _mm_mask_scalef_sd(__W, __U, __A, __B);
}

__m128d test_mm_mask_scalef_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_scalef_round_sd
  // CHECK: @llvm.x86.avx512.mask.scalef.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 %{{.*}}, i32 11)
    return _mm_mask_scalef_round_sd(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_scalef_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_scalef_sd
  // CHECK: @llvm.x86.avx512.mask.scalef.sd
    return _mm_maskz_scalef_sd(__U, __A, __B);
}

__m128d test_mm_maskz_scalef_round_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_scalef_round_sd
  // CHECK: @llvm.x86.avx512.mask.scalef.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 %{{.*}}, i32 11)
    return _mm_maskz_scalef_round_sd(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_scalef_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_scalef_round_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 -1, i32 11)
  return _mm_scalef_round_ss(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_scalef_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_scalef_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss
  return _mm_scalef_ss(__A, __B); 
}

__m128 test_mm_mask_scalef_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_scalef_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss
    return _mm_mask_scalef_ss(__W, __U, __A, __B);
}

__m128 test_mm_mask_scalef_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_scalef_round_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 %{{.*}}, i32 11)
    return _mm_mask_scalef_round_ss(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_scalef_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_scalef_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss
    return _mm_maskz_scalef_ss(__U, __A, __B);
}

__m128 test_mm_maskz_scalef_round_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_scalef_round_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 %{{.*}}, i32 11)
    return _mm_maskz_scalef_round_ss(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_srai_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_srai_epi32
  // CHECK: @llvm.x86.avx512.psrai.d.512
  return _mm512_srai_epi32(__A, 5); 
}

__m512i test_mm512_srai_epi32_2(__m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_srai_epi32_2
  // CHECK: @llvm.x86.avx512.psrai.d.512
  return _mm512_srai_epi32(__A, __B); 
}

__m512i test_mm512_mask_srai_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_srai_epi32
  // CHECK: @llvm.x86.avx512.psrai.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_srai_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_mask_srai_epi32_2(__m512i __W, __mmask16 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_mask_srai_epi32_2
  // CHECK: @llvm.x86.avx512.psrai.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_srai_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srai_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_srai_epi32
  // CHECK: @llvm.x86.avx512.psrai.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_srai_epi32(__U, __A, 5); 
}

__m512i test_mm512_maskz_srai_epi32_2(__mmask16 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_maskz_srai_epi32_2
  // CHECK: @llvm.x86.avx512.psrai.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_srai_epi32(__U, __A, __B); 
}

__m512i test_mm512_srai_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_srai_epi64
  // CHECK: @llvm.x86.avx512.psrai.q.512
  return _mm512_srai_epi64(__A, 5); 
}

__m512i test_mm512_srai_epi64_2(__m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_srai_epi64_2
  // CHECK: @llvm.x86.avx512.psrai.q.512
  return _mm512_srai_epi64(__A, __B); 
}

__m512i test_mm512_mask_srai_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_srai_epi64
  // CHECK: @llvm.x86.avx512.psrai.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_srai_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_mask_srai_epi64_2(__m512i __W, __mmask8 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_mask_srai_epi64_2
  // CHECK: @llvm.x86.avx512.psrai.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_srai_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srai_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_srai_epi64
  // CHECK: @llvm.x86.avx512.psrai.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_srai_epi64(__U, __A, 5); 
}

__m512i test_mm512_maskz_srai_epi64_2(__mmask8 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm512_maskz_srai_epi64_2
  // CHECK: @llvm.x86.avx512.psrai.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_srai_epi64(__U, __A, __B); 
}

__m512i test_mm512_sll_epi32(__m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_sll_epi32
  // CHECK: @llvm.x86.avx512.psll.d.512
  return _mm512_sll_epi32(__A, __B); 
}

__m512i test_mm512_mask_sll_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_mask_sll_epi32
  // CHECK: @llvm.x86.avx512.psll.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_sll_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sll_epi32(__mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_maskz_sll_epi32
  // CHECK: @llvm.x86.avx512.psll.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_sll_epi32(__U, __A, __B); 
}

__m512i test_mm512_sll_epi64(__m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_sll_epi64
  // CHECK: @llvm.x86.avx512.psll.q.512
  return _mm512_sll_epi64(__A, __B); 
}

__m512i test_mm512_mask_sll_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_mask_sll_epi64
  // CHECK: @llvm.x86.avx512.psll.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_sll_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sll_epi64(__mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_maskz_sll_epi64
  // CHECK: @llvm.x86.avx512.psll.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_sll_epi64(__U, __A, __B); 
}

__m512i test_mm512_sllv_epi32(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_sllv_epi32
  // CHECK: @llvm.x86.avx512.psllv.d.512
  return _mm512_sllv_epi32(__X, __Y); 
}

__m512i test_mm512_mask_sllv_epi32(__m512i __W, __mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_sllv_epi32
  // CHECK: @llvm.x86.avx512.psllv.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_sllv_epi32(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_sllv_epi32(__mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_maskz_sllv_epi32
  // CHECK: @llvm.x86.avx512.psllv.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_sllv_epi32(__U, __X, __Y); 
}

__m512i test_mm512_sllv_epi64(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_sllv_epi64
  // CHECK: @llvm.x86.avx512.psllv.q.512
  return _mm512_sllv_epi64(__X, __Y); 
}

__m512i test_mm512_mask_sllv_epi64(__m512i __W, __mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_sllv_epi64
  // CHECK: @llvm.x86.avx512.psllv.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_sllv_epi64(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_sllv_epi64(__mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_maskz_sllv_epi64
  // CHECK: @llvm.x86.avx512.psllv.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_sllv_epi64(__U, __X, __Y); 
}

__m512i test_mm512_sra_epi32(__m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_sra_epi32
  // CHECK: @llvm.x86.avx512.psra.d.512
  return _mm512_sra_epi32(__A, __B); 
}

__m512i test_mm512_mask_sra_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_mask_sra_epi32
  // CHECK: @llvm.x86.avx512.psra.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_sra_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sra_epi32(__mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_maskz_sra_epi32
  // CHECK: @llvm.x86.avx512.psra.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_sra_epi32(__U, __A, __B); 
}

__m512i test_mm512_sra_epi64(__m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_sra_epi64
  // CHECK: @llvm.x86.avx512.psra.q.512
  return _mm512_sra_epi64(__A, __B); 
}

__m512i test_mm512_mask_sra_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_mask_sra_epi64
  // CHECK: @llvm.x86.avx512.psra.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_sra_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sra_epi64(__mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_maskz_sra_epi64
  // CHECK: @llvm.x86.avx512.psra.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_sra_epi64(__U, __A, __B); 
}

__m512i test_mm512_srav_epi32(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_srav_epi32
  // CHECK: @llvm.x86.avx512.psrav.d.512
  return _mm512_srav_epi32(__X, __Y); 
}

__m512i test_mm512_mask_srav_epi32(__m512i __W, __mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_srav_epi32
  // CHECK: @llvm.x86.avx512.psrav.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_srav_epi32(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_srav_epi32(__mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_maskz_srav_epi32
  // CHECK: @llvm.x86.avx512.psrav.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_srav_epi32(__U, __X, __Y); 
}

__m512i test_mm512_srav_epi64(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_srav_epi64
  // CHECK: @llvm.x86.avx512.psrav.q.512
  return _mm512_srav_epi64(__X, __Y); 
}

__m512i test_mm512_mask_srav_epi64(__m512i __W, __mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_srav_epi64
  // CHECK: @llvm.x86.avx512.psrav.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_srav_epi64(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_srav_epi64(__mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_maskz_srav_epi64
  // CHECK: @llvm.x86.avx512.psrav.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_srav_epi64(__U, __X, __Y); 
}

__m512i test_mm512_srl_epi32(__m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_srl_epi32
  // CHECK: @llvm.x86.avx512.psrl.d.512
  return _mm512_srl_epi32(__A, __B); 
}

__m512i test_mm512_mask_srl_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_mask_srl_epi32
  // CHECK: @llvm.x86.avx512.psrl.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_srl_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srl_epi32(__mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_maskz_srl_epi32
  // CHECK: @llvm.x86.avx512.psrl.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_srl_epi32(__U, __A, __B); 
}

__m512i test_mm512_srl_epi64(__m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_srl_epi64
  // CHECK: @llvm.x86.avx512.psrl.q.512
  return _mm512_srl_epi64(__A, __B); 
}

__m512i test_mm512_mask_srl_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_mask_srl_epi64
  // CHECK: @llvm.x86.avx512.psrl.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_srl_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srl_epi64(__mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_maskz_srl_epi64
  // CHECK: @llvm.x86.avx512.psrl.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_srl_epi64(__U, __A, __B); 
}

__m512i test_mm512_srlv_epi32(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_srlv_epi32
  // CHECK: @llvm.x86.avx512.psrlv.d.512
  return _mm512_srlv_epi32(__X, __Y); 
}

__m512i test_mm512_mask_srlv_epi32(__m512i __W, __mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_srlv_epi32
  // CHECK: @llvm.x86.avx512.psrlv.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_srlv_epi32(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_srlv_epi32(__mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_maskz_srlv_epi32
  // CHECK: @llvm.x86.avx512.psrlv.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_srlv_epi32(__U, __X, __Y); 
}

__m512i test_mm512_srlv_epi64(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_srlv_epi64
  // CHECK: @llvm.x86.avx512.psrlv.q.512
  return _mm512_srlv_epi64(__X, __Y); 
}

__m512i test_mm512_mask_srlv_epi64(__m512i __W, __mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_srlv_epi64
  // CHECK: @llvm.x86.avx512.psrlv.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_srlv_epi64(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_srlv_epi64(__mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_maskz_srlv_epi64
  // CHECK: @llvm.x86.avx512.psrlv.q.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_srlv_epi64(__U, __X, __Y); 
}

__m512i test_mm512_ternarylogic_epi32(__m512i __A, __m512i __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.pternlog.d.512({{.*}}, i32 240)
  return _mm512_ternarylogic_epi32(__A, __B, __C, _MM_TERNLOG_A);
}

__m512i test_mm512_mask_ternarylogic_epi32(__m512i __A, __mmask16 __U, __m512i __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_mask_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.pternlog.d.512({{.*}}, i32 204)
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_ternarylogic_epi32(__A, __U, __B, __C, _MM_TERNLOG_B);
}

__m512i test_mm512_maskz_ternarylogic_epi32(__mmask16 __U, __m512i __A, __m512i __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_maskz_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.pternlog.d.512({{.*}}, i32 170)
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> zeroinitializer
  return _mm512_maskz_ternarylogic_epi32(__U, __A, __B, __C, _MM_TERNLOG_C);
}

__m512i test_mm512_ternarylogic_epi64(__m512i __A, __m512i __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.pternlog.q.512({{.*}}, i32 192)
  return _mm512_ternarylogic_epi64(__A, __B, __C, _MM_TERNLOG_A & _MM_TERNLOG_B);
}

__m512i test_mm512_mask_ternarylogic_epi64(__m512i __A, __mmask8 __U, __m512i __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_mask_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.pternlog.q.512({{.*}}, i32 238)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_ternarylogic_epi64(__A, __U, __B, __C, _MM_TERNLOG_B | _MM_TERNLOG_C);
}

__m512i test_mm512_maskz_ternarylogic_epi64(__mmask8 __U, __m512i __A, __m512i __B, __m512i __C) {
  // CHECK-LABEL: test_mm512_maskz_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.pternlog.q.512({{.*}}, i32 111)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> zeroinitializer
  return _mm512_maskz_ternarylogic_epi64(__U, __A, __B, __C, ~_MM_TERNLOG_A | (_MM_TERNLOG_B ^ _MM_TERNLOG_C));
}

__m512 test_mm512_shuffle_f32x4(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_shuffle_f32x4
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 16, i32 17, i32 18, i32 19>
  return _mm512_shuffle_f32x4(__A, __B, 4); 
}

__m512 test_mm512_mask_shuffle_f32x4(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_shuffle_f32x4
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 16, i32 17, i32 18, i32 19>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_shuffle_f32x4(__W, __U, __A, __B, 4); 
}

__m512 test_mm512_maskz_shuffle_f32x4(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_shuffle_f32x4
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 16, i32 17, i32 18, i32 19>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_shuffle_f32x4(__U, __A, __B, 4); 
}

__m512d test_mm512_shuffle_f64x2(__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_shuffle_f64x2
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 8, i32 9>
  return _mm512_shuffle_f64x2(__A, __B, 4); 
}

__m512d test_mm512_mask_shuffle_f64x2(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_shuffle_f64x2
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 8, i32 9>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_shuffle_f64x2(__W, __U, __A, __B, 4); 
}

__m512d test_mm512_maskz_shuffle_f64x2(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_shuffle_f64x2
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 8, i32 9>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_shuffle_f64x2(__U, __A, __B, 4); 
}

__m512i test_mm512_shuffle_i32x4(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shuffle_i32x4
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 16, i32 17, i32 18, i32 19>
  return _mm512_shuffle_i32x4(__A, __B, 4); 
}

__m512i test_mm512_mask_shuffle_i32x4(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shuffle_i32x4
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 16, i32 17, i32 18, i32 19>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shuffle_i32x4(__W, __U, __A, __B, 4); 
}

__m512i test_mm512_maskz_shuffle_i32x4(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shuffle_i32x4
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 16, i32 17, i32 18, i32 19>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shuffle_i32x4(__U, __A, __B, 4); 
}

__m512i test_mm512_shuffle_i64x2(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_shuffle_i64x2
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 8, i32 9>
  return _mm512_shuffle_i64x2(__A, __B, 4); 
}

__m512i test_mm512_mask_shuffle_i64x2(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_shuffle_i64x2
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 8, i32 9>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shuffle_i64x2(__W, __U, __A, __B, 4); 
}

__m512i test_mm512_maskz_shuffle_i64x2(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_shuffle_i64x2
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 8, i32 9>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shuffle_i64x2(__U, __A, __B, 4); 
}

__m512d test_mm512_shuffle_pd(__m512d __M, __m512d __V) {
  // CHECK-LABEL: test_mm512_shuffle_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 3, i32 10, i32 4, i32 12, i32 6, i32 14>
  return _mm512_shuffle_pd(__M, __V, 4); 
}

__m512d test_mm512_mask_shuffle_pd(__m512d __W, __mmask8 __U, __m512d __M, __m512d __V) {
  // CHECK-LABEL: test_mm512_mask_shuffle_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 3, i32 10, i32 4, i32 12, i32 6, i32 14>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_shuffle_pd(__W, __U, __M, __V, 4); 
}

__m512d test_mm512_maskz_shuffle_pd(__mmask8 __U, __m512d __M, __m512d __V) {
  // CHECK-LABEL: test_mm512_maskz_shuffle_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 3, i32 10, i32 4, i32 12, i32 6, i32 14>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_shuffle_pd(__U, __M, __V, 4); 
}

__m512 test_mm512_shuffle_ps(__m512 __M, __m512 __V) {
  // CHECK-LABEL: test_mm512_shuffle_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 16, i32 16, i32 4, i32 5, i32 20, i32 20, i32 8, i32 9, i32 24, i32 24, i32 12, i32 13, i32 28, i32 28>
  return _mm512_shuffle_ps(__M, __V, 4); 
}

__m512 test_mm512_mask_shuffle_ps(__m512 __W, __mmask16 __U, __m512 __M, __m512 __V) {
  // CHECK-LABEL: test_mm512_mask_shuffle_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 16, i32 16, i32 4, i32 5, i32 20, i32 20, i32 8, i32 9, i32 24, i32 24, i32 12, i32 13, i32 28, i32 28>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_shuffle_ps(__W, __U, __M, __V, 4); 
}

__m512 test_mm512_maskz_shuffle_ps(__mmask16 __U, __m512 __M, __m512 __V) {
  // CHECK-LABEL: test_mm512_maskz_shuffle_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 16, i32 16, i32 4, i32 5, i32 20, i32 20, i32 8, i32 9, i32 24, i32 24, i32 12, i32 13, i32 28, i32 28>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_shuffle_ps(__U, __M, __V, 4); 
}

__m128d test_mm_sqrt_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_sqrt_round_sd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.avx512.mask.sqrt.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 -1, i32 11)
  return _mm_sqrt_round_sd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_sqrt_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_sqrt_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: call double @llvm.sqrt.f64(double %{{.*}})
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 {{.*}}, double {{.*}}, double {{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double {{.*}}, i64 0
  return _mm_mask_sqrt_sd(__W,__U,__A,__B);
}

__m128d test_mm_mask_sqrt_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_sqrt_round_sd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.avx512.mask.sqrt.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 %{{.*}}, i32 11)
  return _mm_mask_sqrt_round_sd(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_sqrt_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_sqrt_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: call double @llvm.sqrt.f64(double %{{.*}})
  // CHECK-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 {{.*}}, double {{.*}}, double {{.*}}
  // CHECK-NEXT: insertelement <2 x double> %{{.*}}, double {{.*}}, i64 0
  return _mm_maskz_sqrt_sd(__U,__A,__B);
}

__m128d test_mm_maskz_sqrt_round_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_sqrt_round_sd
  // CHECK: call {{.*}}<2 x double> @llvm.x86.avx512.mask.sqrt.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 %{{.*}}, i32 11)
  return _mm_maskz_sqrt_round_sd(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_sqrt_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_sqrt_round_ss
  // CHECK: call {{.*}}<4 x float> @llvm.x86.avx512.mask.sqrt.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 -1, i32 11)
  return _mm_sqrt_round_ss(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_sqrt_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_sqrt_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: call float @llvm.sqrt.f32(float %{{.*}})
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 {{.*}}, float {{.*}}, float {{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float {{.*}}, i64 0
  return _mm_mask_sqrt_ss(__W,__U,__A,__B);
}

__m128 test_mm_mask_sqrt_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_sqrt_round_ss
  // CHECK: call {{.*}}<4 x float> @llvm.x86.avx512.mask.sqrt.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 {{.*}}, i32 11)
  return _mm_mask_sqrt_round_ss(__W,__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_sqrt_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_sqrt_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: call float @llvm.sqrt.f32(float %{{.*}})
  // CHECK-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: select i1 {{.*}}, float {{.*}}, float {{.*}}
  // CHECK-NEXT: insertelement <4 x float> %{{.*}}, float {{.*}}, i64 0
  return _mm_maskz_sqrt_ss(__U,__A,__B);
}

__m128 test_mm_maskz_sqrt_round_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_sqrt_round_ss
  // CHECK: call {{.*}}<4 x float> @llvm.x86.avx512.mask.sqrt.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 {{.*}}, i32 11)
  return _mm_maskz_sqrt_round_ss(__U,__A,__B,_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_broadcast_f32x4(float const* __A) {
  // CHECK-LABEL: test_mm512_broadcast_f32x4
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  return _mm512_broadcast_f32x4(_mm_loadu_ps(__A)); 
}

__m512 test_mm512_mask_broadcast_f32x4(__m512 __O, __mmask16 __M, float const* __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_f32x4
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_broadcast_f32x4(__O, __M, _mm_loadu_ps(__A)); 
}

__m512 test_mm512_maskz_broadcast_f32x4(__mmask16 __M, float const* __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_f32x4
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_broadcast_f32x4(__M, _mm_loadu_ps(__A)); 
}

__m512d test_mm512_broadcast_f64x4(double const* __A) {
  // CHECK-LABEL: test_mm512_broadcast_f64x4
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  return _mm512_broadcast_f64x4(_mm256_loadu_pd(__A)); 
}

__m512d test_mm512_mask_broadcast_f64x4(__m512d __O, __mmask8 __M, double const* __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_f64x4
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_broadcast_f64x4(__O, __M, _mm256_loadu_pd(__A)); 
}

__m512d test_mm512_maskz_broadcast_f64x4(__mmask8 __M, double const* __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_f64x4
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_broadcast_f64x4(__M, _mm256_loadu_pd(__A)); 
}

__m512i test_mm512_broadcast_i32x4(__m128i const* __A) {
  // CHECK-LABEL: test_mm512_broadcast_i32x4
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  return _mm512_broadcast_i32x4(_mm_loadu_si128(__A)); 
}

__m512i test_mm512_mask_broadcast_i32x4(__m512i __O, __mmask16 __M, __m128i const* __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_i32x4
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_broadcast_i32x4(__O, __M, _mm_loadu_si128(__A)); 
}

__m512i test_mm512_maskz_broadcast_i32x4(__mmask16 __M, __m128i const* __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_i32x4
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_broadcast_i32x4(__M, _mm_loadu_si128(__A)); 
}

__m512i test_mm512_broadcast_i64x4(__m256i const* __A) {
  // CHECK-LABEL: test_mm512_broadcast_i64x4
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  return _mm512_broadcast_i64x4(_mm256_loadu_si256(__A)); 
}

__m512i test_mm512_mask_broadcast_i64x4(__m512i __O, __mmask8 __M, __m256i const* __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_i64x4
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_broadcast_i64x4(__O, __M, _mm256_loadu_si256(__A)); 
}

__m512i test_mm512_maskz_broadcast_i64x4(__mmask8 __M, __m256i const* __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_i64x4
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_broadcast_i64x4(__M, _mm256_loadu_si256(__A)); 
}

__m512d test_mm512_broadcastsd_pd(__m128d __A) {
  // CHECK-LABEL: test_mm512_broadcastsd_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <8 x i32> zeroinitializer
  return _mm512_broadcastsd_pd(__A);
}

__m512d test_mm512_mask_broadcastsd_pd(__m512d __O, __mmask8 __M, __m128d __A) {
  // CHECK-LABEL: test_mm512_mask_broadcastsd_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_broadcastsd_pd(__O, __M, __A);
}

__m512d test_mm512_maskz_broadcastsd_pd(__mmask8 __M, __m128d __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcastsd_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_broadcastsd_pd(__M, __A);
}

__m512 test_mm512_broadcastss_ps(__m128 __A) {
  // CHECK-LABEL: test_mm512_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> zeroinitializer
  return _mm512_broadcastss_ps(__A);
}

__m512 test_mm512_mask_broadcastss_ps(__m512 __O, __mmask16 __M, __m128 __A) {
  // CHECK-LABEL: test_mm512_mask_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_broadcastss_ps(__O, __M, __A);
}

__m512 test_mm512_maskz_broadcastss_ps(__mmask16 __M, __m128 __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_broadcastss_ps(__M, __A);
}

__m512i test_mm512_broadcastd_epi32(__m128i __A) {
  // CHECK-LABEL: test_mm512_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> zeroinitializer
  return _mm512_broadcastd_epi32(__A);
}

__m512i test_mm512_mask_broadcastd_epi32(__m512i __O, __mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm512_mask_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_broadcastd_epi32(__O, __M, __A);
}

__m512i test_mm512_maskz_broadcastd_epi32(__mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_broadcastd_epi32(__M, __A);
}

__m512i test_mm512_broadcastq_epi64(__m128i __A) {
  // CHECK-LABEL: test_mm512_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <8 x i32> zeroinitializer
  return _mm512_broadcastq_epi64(__A);
}

__m512i test_mm512_mask_broadcastq_epi64(__m512i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm512_mask_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_broadcastq_epi64(__O, __M, __A);
}

__m512i test_mm512_maskz_broadcastq_epi64(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_broadcastq_epi64(__M, __A);
}

__m128i test_mm512_cvtsepi32_epi8(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.512
  return _mm512_cvtsepi32_epi8(__A); 
}

__m128i test_mm512_mask_cvtsepi32_epi8(__m128i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.512
  return _mm512_mask_cvtsepi32_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtsepi32_epi8(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.512
  return _mm512_maskz_cvtsepi32_epi8(__M, __A); 
}

void test_mm512_mask_cvtsepi32_storeu_epi8(void * __P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.mem.512
  return _mm512_mask_cvtsepi32_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtsepi32_epi16(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.512
  return _mm512_cvtsepi32_epi16(__A); 
}

__m256i test_mm512_mask_cvtsepi32_epi16(__m256i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.512
  return _mm512_mask_cvtsepi32_epi16(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtsepi32_epi16(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.512
  return _mm512_maskz_cvtsepi32_epi16(__M, __A); 
}

void test_mm512_mask_cvtsepi32_storeu_epi16(void *__P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.mem.512
  return _mm512_mask_cvtsepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtsepi64_epi8(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.512
  return _mm512_cvtsepi64_epi8(__A); 
}

__m128i test_mm512_mask_cvtsepi64_epi8(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.512
  return _mm512_mask_cvtsepi64_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtsepi64_epi8(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.512
  return _mm512_maskz_cvtsepi64_epi8(__M, __A); 
}

void test_mm512_mask_cvtsepi64_storeu_epi8(void * __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.mem.512
  return _mm512_mask_cvtsepi64_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtsepi64_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.512
  return _mm512_cvtsepi64_epi32(__A); 
}

__m256i test_mm512_mask_cvtsepi64_epi32(__m256i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.512
  return _mm512_mask_cvtsepi64_epi32(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtsepi64_epi32(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.512
  return _mm512_maskz_cvtsepi64_epi32(__M, __A); 
}

void test_mm512_mask_cvtsepi64_storeu_epi32(void *__P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.mem.512
  return _mm512_mask_cvtsepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm512_cvtsepi64_epi16(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.512
  return _mm512_cvtsepi64_epi16(__A); 
}

__m128i test_mm512_mask_cvtsepi64_epi16(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.512
  return _mm512_mask_cvtsepi64_epi16(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtsepi64_epi16(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.512
  return _mm512_maskz_cvtsepi64_epi16(__M, __A); 
}

void test_mm512_mask_cvtsepi64_storeu_epi16(void * __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtsepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.mem.512
  return _mm512_mask_cvtsepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtusepi32_epi8(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.512
  return _mm512_cvtusepi32_epi8(__A); 
}

__m128i test_mm512_mask_cvtusepi32_epi8(__m128i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.512
  return _mm512_mask_cvtusepi32_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtusepi32_epi8(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.512
  return _mm512_maskz_cvtusepi32_epi8(__M, __A); 
}

void test_mm512_mask_cvtusepi32_storeu_epi8(void * __P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.mem.512
  return _mm512_mask_cvtusepi32_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtusepi32_epi16(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.512
  return _mm512_cvtusepi32_epi16(__A); 
}

__m256i test_mm512_mask_cvtusepi32_epi16(__m256i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.512
  return _mm512_mask_cvtusepi32_epi16(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtusepi32_epi16(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.512
  return _mm512_maskz_cvtusepi32_epi16(__M, __A); 
}

void test_mm512_mask_cvtusepi32_storeu_epi16(void *__P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.mem.512
  return _mm512_mask_cvtusepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtusepi64_epi8(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.512
  return _mm512_cvtusepi64_epi8(__A); 
}

__m128i test_mm512_mask_cvtusepi64_epi8(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.512
  return _mm512_mask_cvtusepi64_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtusepi64_epi8(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.512
  return _mm512_maskz_cvtusepi64_epi8(__M, __A); 
}

void test_mm512_mask_cvtusepi64_storeu_epi8(void * __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.mem.512
  return _mm512_mask_cvtusepi64_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtusepi64_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.512
  return _mm512_cvtusepi64_epi32(__A); 
}

__m256i test_mm512_mask_cvtusepi64_epi32(__m256i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.512
  return _mm512_mask_cvtusepi64_epi32(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtusepi64_epi32(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.512
  return _mm512_maskz_cvtusepi64_epi32(__M, __A); 
}

void test_mm512_mask_cvtusepi64_storeu_epi32(void* __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.mem.512
  return _mm512_mask_cvtusepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm512_cvtusepi64_epi16(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.512
  return _mm512_cvtusepi64_epi16(__A); 
}

__m128i test_mm512_mask_cvtusepi64_epi16(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.512
  return _mm512_mask_cvtusepi64_epi16(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtusepi64_epi16(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.512
  return _mm512_maskz_cvtusepi64_epi16(__M, __A); 
}

void test_mm512_mask_cvtusepi64_storeu_epi16(void *__P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtusepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.mem.512
  return _mm512_mask_cvtusepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtepi32_epi8(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtepi32_epi8
  // CHECK: trunc <16 x i32> %{{.*}} to <16 x i8>
  return _mm512_cvtepi32_epi8(__A); 
}

__m128i test_mm512_mask_cvtepi32_epi8(__m128i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.512
  return _mm512_mask_cvtepi32_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtepi32_epi8(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.512
  return _mm512_maskz_cvtepi32_epi8(__M, __A); 
}

void test_mm512_mask_cvtepi32_storeu_epi8(void * __P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.mem.512
  return _mm512_mask_cvtepi32_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtepi32_epi16(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtepi32_epi16
  // CHECK: trunc <16 x i32> %{{.*}} to <16 x i16>
  return _mm512_cvtepi32_epi16(__A); 
}

__m256i test_mm512_mask_cvtepi32_epi16(__m256i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.512
  return _mm512_mask_cvtepi32_epi16(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtepi32_epi16(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.512
  return _mm512_maskz_cvtepi32_epi16(__M, __A); 
}

void test_mm512_mask_cvtepi32_storeu_epi16(void * __P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.mem.512
  return _mm512_mask_cvtepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtepi64_epi8(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.512
  return _mm512_cvtepi64_epi8(__A); 
}

__m128i test_mm512_mask_cvtepi64_epi8(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.512
  return _mm512_mask_cvtepi64_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtepi64_epi8(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.512
  return _mm512_maskz_cvtepi64_epi8(__M, __A); 
}

void test_mm512_mask_cvtepi64_storeu_epi8(void * __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.mem.512
  return _mm512_mask_cvtepi64_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtepi64_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtepi64_epi32
  // CHECK: trunc <8 x i64> %{{.*}} to <8 x i32>
  return _mm512_cvtepi64_epi32(__A); 
}

__m256i test_mm512_mask_cvtepi64_epi32(__m256i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi64_epi32
  // CHECK: trunc <8 x i64> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm512_mask_cvtepi64_epi32(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtepi64_epi32(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi64_epi32
  // CHECK: trunc <8 x i64> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm512_maskz_cvtepi64_epi32(__M, __A); 
}

void test_mm512_mask_cvtepi64_storeu_epi32(void* __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.mem.512
  return _mm512_mask_cvtepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm512_cvtepi64_epi16(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtepi64_epi16
  // CHECK: trunc <8 x i64> %{{.*}} to <8 x i16>
  return _mm512_cvtepi64_epi16(__A); 
}

__m128i test_mm512_mask_cvtepi64_epi16(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.512
  return _mm512_mask_cvtepi64_epi16(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtepi64_epi16(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.512
  return _mm512_maskz_cvtepi64_epi16(__M, __A); 
}

void test_mm512_mask_cvtepi64_storeu_epi16(void *__P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.mem.512
  return _mm512_mask_cvtepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_extracti32x4_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_extracti32x4_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  return _mm512_extracti32x4_epi32(__A, 3); 
}

__m128i test_mm512_mask_extracti32x4_epi32(__m128i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_extracti32x4_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm512_mask_extracti32x4_epi32(__W, __U, __A, 3); 
}

__m128i test_mm512_maskz_extracti32x4_epi32(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_extracti32x4_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm512_maskz_extracti32x4_epi32(__U, __A, 3); 
}

__m256i test_mm512_extracti64x4_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_extracti64x4_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm512_extracti64x4_epi64(__A, 1); 
}

__m256i test_mm512_mask_extracti64x4_epi64(__m256i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_extracti64x4_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm512_mask_extracti64x4_epi64(__W, __U, __A, 1); 
}

__m256i test_mm512_maskz_extracti64x4_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_extracti64x4_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm512_maskz_extracti64x4_epi64(__U, __A, 1); 
}

__m512d test_mm512_insertf64x4(__m512d __A, __m256d __B) {
  // CHECK-LABEL: test_mm512_insertf64x4
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm512_insertf64x4(__A, __B, 1);
}

__m512d test_mm512_mask_insertf64x4(__m512d __W, __mmask8 __U, __m512d __A, __m256d __B) {
  // CHECK-LABEL: test_mm512_mask_insertf64x4
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_insertf64x4(__W, __U, __A, __B, 1); 
}

__m512d test_mm512_maskz_insertf64x4(__mmask8 __U, __m512d __A, __m256d __B) {
  // CHECK-LABEL: test_mm512_maskz_insertf64x4
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_insertf64x4(__U, __A, __B, 1); 
}

__m512i test_mm512_inserti64x4(__m512i __A, __m256i __B) {
  // CHECK-LABEL: test_mm512_inserti64x4
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm512_inserti64x4(__A, __B, 1); 
}

__m512i test_mm512_mask_inserti64x4(__m512i __W, __mmask8 __U, __m512i __A, __m256i __B) {
  // CHECK-LABEL: test_mm512_mask_inserti64x4
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_inserti64x4(__W, __U, __A, __B, 1); 
}

__m512i test_mm512_maskz_inserti64x4(__mmask8 __U, __m512i __A, __m256i __B) {
  // CHECK-LABEL: test_mm512_maskz_inserti64x4
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_inserti64x4(__U, __A, __B, 1); 
}

__m512 test_mm512_insertf32x4(__m512 __A, __m128 __B) {
  // CHECK-LABEL: test_mm512_insertf32x4
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_insertf32x4(__A, __B, 1);
}

__m512 test_mm512_mask_insertf32x4(__m512 __W, __mmask16 __U, __m512 __A, __m128 __B) {
  // CHECK-LABEL: test_mm512_mask_insertf32x4
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_insertf32x4(__W, __U, __A, __B, 1); 
}

__m512 test_mm512_maskz_insertf32x4(__mmask16 __U, __m512 __A, __m128 __B) {
  // CHECK-LABEL: test_mm512_maskz_insertf32x4
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_insertf32x4(__U, __A, __B, 1); 
}

__m512i test_mm512_inserti32x4(__m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_inserti32x4
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_inserti32x4(__A, __B, 1); 
}

__m512i test_mm512_mask_inserti32x4(__m512i __W, __mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_mask_inserti32x4
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_inserti32x4(__W, __U, __A, __B, 1); 
}

__m512i test_mm512_maskz_inserti32x4(__mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_maskz_inserti32x4
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_inserti32x4(__U, __A, __B, 1); 
}

__m512d test_mm512_getmant_round_pd(__m512d __A) {
  // CHECK-LABEL: test_mm512_getmant_round_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_getmant_round_pd(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_getmant_round_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_getmant_round_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_mask_getmant_round_pd(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_getmant_round_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_getmant_round_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_maskz_getmant_round_pd(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_getmant_pd(__m512d __A) {
  // CHECK-LABEL: test_mm512_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_getmant_pd(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512d test_mm512_mask_getmant_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_mask_getmant_pd(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512d test_mm512_maskz_getmant_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_maskz_getmant_pd(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512 test_mm512_getmant_round_ps(__m512 __A) {
  // CHECK-LABEL: test_mm512_getmant_round_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_getmant_round_ps(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_getmant_round_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_getmant_round_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_mask_getmant_round_ps(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_getmant_round_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_getmant_round_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_maskz_getmant_round_ps(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_getmant_ps(__m512 __A) {
  // CHECK-LABEL: test_mm512_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_getmant_ps(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512 test_mm512_mask_getmant_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_mask_getmant_ps(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512 test_mm512_maskz_getmant_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_maskz_getmant_ps(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512d test_mm512_getexp_round_pd(__m512d __A) {
  // CHECK-LABEL: test_mm512_getexp_round_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_getexp_round_pd(__A, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_getexp_round_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_getexp_round_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_mask_getexp_round_pd(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_getexp_round_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_getexp_round_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_maskz_getexp_round_pd(__U, __A, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_getexp_pd(__m512d __A) {
  // CHECK-LABEL: test_mm512_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_getexp_pd(__A); 
}

__m512d test_mm512_mask_getexp_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_mask_getexp_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_getexp_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_maskz_getexp_pd(__U, __A); 
}

__m512 test_mm512_getexp_round_ps(__m512 __A) {
  // CHECK-LABEL: test_mm512_getexp_round_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_getexp_round_ps(__A, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_getexp_round_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_getexp_round_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_mask_getexp_round_ps(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_getexp_round_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_getexp_round_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_maskz_getexp_round_ps(__U, __A, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_getexp_ps(__m512 __A) {
  // CHECK-LABEL: test_mm512_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_getexp_ps(__A); 
}

__m512 test_mm512_mask_getexp_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_mask_getexp_ps(__W, __U, __A); 
}

__m512 test_mm512_maskz_getexp_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_maskz_getexp_ps(__U, __A); 
}

__m256 test_mm512_i64gather_ps(__m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i64gather_ps
  // CHECK: @llvm.x86.avx512.mask.gather.qps.512
  return _mm512_i64gather_ps(__index, __addr, 2); 
}

__m256 test_mm512_mask_i64gather_ps(__m256 __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i64gather_ps
  // CHECK: @llvm.x86.avx512.mask.gather.qps.512
  return _mm512_mask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm512_i64gather_epi32(__m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i64gather_epi32
  // CHECK: @llvm.x86.avx512.mask.gather.qpi.512
  return _mm512_i64gather_epi32(__index, __addr, 2); 
}

__m256i test_mm512_mask_i64gather_epi32(__m256i __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i64gather_epi32
  // CHECK: @llvm.x86.avx512.mask.gather.qpi.512
  return _mm512_mask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m512d test_mm512_i64gather_pd(__m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i64gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather.qpd.512
  return _mm512_i64gather_pd(__index, __addr, 2); 
}

__m512d test_mm512_mask_i64gather_pd(__m512d __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i64gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather.qpd.512
  return _mm512_mask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m512i test_mm512_i64gather_epi64(__m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i64gather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather.qpq.512
  return _mm512_i64gather_epi64(__index, __addr, 2); 
}

__m512i test_mm512_mask_i64gather_epi64(__m512i __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i64gather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather.qpq.512
  return _mm512_mask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m512 test_mm512_i32gather_ps(__m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i32gather_ps
  // CHECK: @llvm.x86.avx512.mask.gather.dps.512
  return _mm512_i32gather_ps(__index, __addr, 2); 
}

__m512 test_mm512_mask_i32gather_ps(__m512 v1_old, __mmask16 __mask, __m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i32gather_ps
  // CHECK: @llvm.x86.avx512.mask.gather.dps.512
  return _mm512_mask_i32gather_ps(v1_old, __mask, __index, __addr, 2); 
}

__m512i test_mm512_i32gather_epi32(__m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i32gather_epi32
  // CHECK: @llvm.x86.avx512.mask.gather.dpi.512
  return _mm512_i32gather_epi32(__index, __addr, 2); 
}

__m512i test_mm512_mask_i32gather_epi32(__m512i __v1_old, __mmask16 __mask, __m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i32gather_epi32
  // CHECK: @llvm.x86.avx512.mask.gather.dpi.512
  return _mm512_mask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m512d test_mm512_i32gather_pd(__m256i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i32gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_i32gather_pd(__index, __addr, 2); 
}

__m512d test_mm512_mask_i32gather_pd(__m512d __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i32gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_mask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m512i test_mm512_i32gather_epi64(__m256i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i32gather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_i32gather_epi64(__index, __addr, 2); 
}

__m512i test_mm512_mask_i32gather_epi64(__m512i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i32gather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_mask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

void test_mm512_i64scatter_ps(void *__addr, __m512i __index, __m256 __v1) {
  // CHECK-LABEL: test_mm512_i64scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scatter.qps.512
  return _mm512_i64scatter_ps(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i64scatter_ps(void *__addr, __mmask8 __mask, __m512i __index, __m256 __v1) {
  // CHECK-LABEL: test_mm512_mask_i64scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scatter.qps.512
  return _mm512_mask_i64scatter_ps(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i64scatter_epi32(void *__addr, __m512i __index, __m256i __v1) {
  // CHECK-LABEL: test_mm512_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scatter.qpi.512
  return _mm512_i64scatter_epi32(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i64scatter_epi32(void *__addr, __mmask8 __mask, __m512i __index, __m256i __v1) {
  // CHECK-LABEL: test_mm512_mask_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scatter.qpi.512
  return _mm512_mask_i64scatter_epi32(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i64scatter_pd(void *__addr, __m512i __index, __m512d __v1) {
  // CHECK-LABEL: test_mm512_i64scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatter.qpd.512
  return _mm512_i64scatter_pd(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i64scatter_pd(void *__addr, __mmask8 __mask, __m512i __index, __m512d __v1) {
  // CHECK-LABEL: test_mm512_mask_i64scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatter.qpd.512
  return _mm512_mask_i64scatter_pd(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i64scatter_epi64(void *__addr, __m512i __index, __m512i __v1) {
  // CHECK-LABEL: test_mm512_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatter.qpq.512
  return _mm512_i64scatter_epi64(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i64scatter_epi64(void *__addr, __mmask8 __mask, __m512i __index, __m512i __v1) {
  // CHECK-LABEL: test_mm512_mask_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatter.qpq.512
  return _mm512_mask_i64scatter_epi64(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i32scatter_ps(void *__addr, __m512i __index, __m512 __v1) {
  // CHECK-LABEL: test_mm512_i32scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scatter.dps.512
  return _mm512_i32scatter_ps(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i32scatter_ps(void *__addr, __mmask16 __mask, __m512i __index, __m512 __v1) {
  // CHECK-LABEL: test_mm512_mask_i32scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scatter.dps.512
  return _mm512_mask_i32scatter_ps(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i32scatter_epi32(void *__addr, __m512i __index, __m512i __v1) {
  // CHECK-LABEL: test_mm512_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scatter.dpi.512
  return _mm512_i32scatter_epi32(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i32scatter_epi32(void *__addr, __mmask16 __mask, __m512i __index, __m512i __v1) {
  // CHECK-LABEL: test_mm512_mask_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scatter.dpi.512
  return _mm512_mask_i32scatter_epi32(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i32scatter_pd(void *__addr, __m256i __index, __m512d __v1) {
  // CHECK-LABEL: test_mm512_i32scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatter.dpd.512
  return _mm512_i32scatter_pd(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i32scatter_pd(void *__addr, __mmask8 __mask, __m256i __index, __m512d __v1) {
  // CHECK-LABEL: test_mm512_mask_i32scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatter.dpd.512
  return _mm512_mask_i32scatter_pd(__addr, __mask, __index, __v1, 2); 
}

void test_mm512_i32scatter_epi64(void *__addr, __m256i __index, __m512i __v1) {
  // CHECK-LABEL: test_mm512_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatter.dpq.512
  return _mm512_i32scatter_epi64(__addr, __index, __v1, 2); 
}

void test_mm512_mask_i32scatter_epi64(void *__addr, __mmask8 __mask, __m256i __index, __m512i __v1) {
  // CHECK-LABEL: test_mm512_mask_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatter.dpq.512
  return _mm512_mask_i32scatter_epi64(__addr, __mask, __index, __v1, 2); 
}

__m128d test_mm_mask_rsqrt14_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_rsqrt14_sd
  // CHECK: @llvm.x86.avx512.rsqrt14.sd
  return _mm_mask_rsqrt14_sd(__W, __U, __A, __B);
}

__m128d test_mm_maskz_rsqrt14_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_rsqrt14_sd
  // CHECK: @llvm.x86.avx512.rsqrt14.sd
  return _mm_maskz_rsqrt14_sd(__U, __A, __B);
}

__m128 test_mm_mask_rsqrt14_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_rsqrt14_ss
  // CHECK: @llvm.x86.avx512.rsqrt14.ss
  return _mm_mask_rsqrt14_ss(__W, __U, __A, __B);
}

__m128 test_mm_maskz_rsqrt14_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_rsqrt14_ss
  // CHECK: @llvm.x86.avx512.rsqrt14.ss
  return _mm_maskz_rsqrt14_ss(__U, __A, __B);
}

__m512d test_mm512_mask_rcp14_pd (__m512d __W, __mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_rcp14_pd 
  // CHECK: @llvm.x86.avx512.rcp14.pd.512
  return _mm512_mask_rcp14_pd (__W,__U,__A);
}

__m512d test_mm512_maskz_rcp14_pd (__mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_maskz_rcp14_pd 
  // CHECK: @llvm.x86.avx512.rcp14.pd.512
  return _mm512_maskz_rcp14_pd (__U,__A);
}

__m512 test_mm512_mask_rcp14_ps (__m512 __W, __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_rcp14_ps 
  // CHECK: @llvm.x86.avx512.rcp14.ps.512
  return _mm512_mask_rcp14_ps (__W,__U,__A);
}

__m512 test_mm512_maskz_rcp14_ps (__mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_rcp14_ps 
  // CHECK: @llvm.x86.avx512.rcp14.ps.512
  return _mm512_maskz_rcp14_ps (__U,__A);
}

__m128d test_mm_mask_rcp14_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_rcp14_sd
  // CHECK: @llvm.x86.avx512.rcp14.sd
  return _mm_mask_rcp14_sd(__W, __U, __A, __B);
}

__m128d test_mm_maskz_rcp14_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_rcp14_sd
  // CHECK: @llvm.x86.avx512.rcp14.sd
  return _mm_maskz_rcp14_sd(__U, __A, __B);
}

__m128 test_mm_mask_rcp14_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_rcp14_ss
  // CHECK: @llvm.x86.avx512.rcp14.ss
  return _mm_mask_rcp14_ss(__W, __U, __A, __B);
}

__m128 test_mm_maskz_rcp14_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_rcp14_ss
  // CHECK: @llvm.x86.avx512.rcp14.ss
  return _mm_maskz_rcp14_ss(__U, __A, __B);
}

__m128d test_mm_mask_getexp_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_getexp_sd
  // CHECK: @llvm.x86.avx512.mask.getexp.sd
  return _mm_mask_getexp_sd(__W, __U, __A, __B);
}

__m128d test_mm_mask_getexp_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_getexp_round_sd
  // CHECK: @llvm.x86.avx512.mask.getexp.sd
  return _mm_mask_getexp_round_sd(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_getexp_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_getexp_sd
  // CHECK: @llvm.x86.avx512.mask.getexp.sd
  return _mm_maskz_getexp_sd(__U, __A, __B);
}

__m128d test_mm_maskz_getexp_round_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_getexp_round_sd
  // CHECK: @llvm.x86.avx512.mask.getexp.sd
  return _mm_maskz_getexp_round_sd(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_getexp_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_getexp_ss
  // CHECK: @llvm.x86.avx512.mask.getexp.ss
  return _mm_mask_getexp_ss(__W, __U, __A, __B);
}

__m128 test_mm_mask_getexp_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_getexp_round_ss
  // CHECK: @llvm.x86.avx512.mask.getexp.ss
  return _mm_mask_getexp_round_ss(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_getexp_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_getexp_ss
  // CHECK: @llvm.x86.avx512.mask.getexp.ss
  return _mm_maskz_getexp_ss(__U, __A, __B);
}

__m128 test_mm_maskz_getexp_round_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_getexp_round_ss
  // CHECK: @llvm.x86.avx512.mask.getexp.ss
  return _mm_maskz_getexp_round_ss(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_getmant_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_getmant_sd
  // CHECK: @llvm.x86.avx512.mask.getmant.sd
  return _mm_mask_getmant_sd(__W, __U, __A, __B, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128d test_mm_mask_getmant_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_getmant_round_sd
  // CHECK: @llvm.x86.avx512.mask.getmant.sd
  return _mm_mask_getmant_round_sd(__W, __U, __A, __B, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_getmant_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_getmant_sd
  // CHECK: @llvm.x86.avx512.mask.getmant.sd
  return _mm_maskz_getmant_sd(__U, __A, __B, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128d test_mm_maskz_getmant_round_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_maskz_getmant_round_sd
  // CHECK: @llvm.x86.avx512.mask.getmant.sd
  return _mm_maskz_getmant_round_sd(__U, __A, __B, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_getmant_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_getmant_ss
  // CHECK: @llvm.x86.avx512.mask.getmant.ss
  return _mm_mask_getmant_ss(__W, __U, __A, __B, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128 test_mm_mask_getmant_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_getmant_round_ss
  // CHECK: @llvm.x86.avx512.mask.getmant.ss
  return _mm_mask_getmant_round_ss(__W, __U, __A, __B, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_getmant_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_getmant_ss
  // CHECK: @llvm.x86.avx512.mask.getmant.ss
  return _mm_maskz_getmant_ss(__U, __A, __B, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128 test_mm_maskz_getmant_round_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_maskz_getmant_round_ss
  // CHECK: @llvm.x86.avx512.mask.getmant.ss
  return _mm_maskz_getmant_round_ss(__U, __A, __B, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_fmadd_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_fmadd_ss
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[A]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_mask_fmadd_ss(__W, __U, __A, __B);
}

__m128 test_mm_fmadd_round_ss(__m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_fmadd_round_ss
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[FMA]], i64 0
  return _mm_fmadd_round_ss(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_fmadd_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_fmadd_round_ss
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[A]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_mask_fmadd_round_ss(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_fmadd_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_maskz_fmadd_ss
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float 0.000000e+00
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_maskz_fmadd_ss(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fmadd_round_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_maskz_fmadd_round_ss
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float 0.000000e+00
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_maskz_fmadd_round_ss(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask3_fmadd_ss(__m128 __W, __m128 __X, __m128 __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fmadd_ss
  // CHECK: [[A:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[C]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGC]], float [[SEL]], i64 0
  return _mm_mask3_fmadd_ss(__W, __X, __Y, __U);
}

__m128 test_mm_mask3_fmadd_round_ss(__m128 __W, __m128 __X, __m128 __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fmadd_round_ss
  // CHECK: [[A:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[C]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGC]], float [[SEL]], i64 0
  return _mm_mask3_fmadd_round_ss(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_fmsub_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_fmsub_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[A]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_mask_fmsub_ss(__W, __U, __A, __B);
}

__m128 test_mm_fmsub_round_ss(__m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_fmsub_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[FMA]], i64 0
  return _mm_fmsub_round_ss(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_fmsub_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_fmsub_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[A]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_mask_fmsub_round_ss(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_fmsub_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_maskz_fmsub_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float 0.000000e+00
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_maskz_fmsub_ss(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fmsub_round_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_maskz_fmsub_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float 0.000000e+00
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_maskz_fmsub_round_ss(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask3_fmsub_ss(__m128 __W, __m128 __X, __m128 __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fmsub_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> [[ORIGC:%.+]]
  // CHECK: [[A:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: [[C2:%.+]] = extractelement <4 x float> [[ORIGC]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[C2]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGC]], float [[SEL]], i64 0
  return _mm_mask3_fmsub_ss(__W, __X, __Y, __U);
}

__m128 test_mm_mask3_fmsub_round_ss(__m128 __W, __m128 __X, __m128 __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fmsub_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> [[ORIGC:%.+]]
  // CHECK: [[A:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: [[C2:%.+]] = extractelement <4 x float> [[ORIGC]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[C2]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGC]], float [[SEL]], i64 0
  return _mm_mask3_fmsub_round_ss(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_fnmadd_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_fnmadd_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[A]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_mask_fnmadd_ss(__W, __U, __A, __B);
}

__m128 test_mm_fnmadd_round_ss(__m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_fnmadd_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[FMA]], i64 0
  return _mm_fnmadd_round_ss(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_fnmadd_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_fnmadd_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[A]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_mask_fnmadd_round_ss(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_fnmadd_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_maskz_fnmadd_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float 0.000000e+00
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_maskz_fnmadd_ss(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fnmadd_round_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_maskz_fnmadd_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float 0.000000e+00
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_maskz_fnmadd_round_ss(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask3_fnmadd_ss(__m128 __W, __m128 __X, __m128 __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fnmadd_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[C]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGC]], float [[SEL]], i64 0
  return _mm_mask3_fnmadd_ss(__W, __X, __Y, __U);
}

__m128 test_mm_mask3_fnmadd_round_ss(__m128 __W, __m128 __X, __m128 __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fnmadd_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[C]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGC]], float [[SEL]], i64 0
  return _mm_mask3_fnmadd_round_ss(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_fnmsub_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_fnmsub_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[A]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_mask_fnmsub_ss(__W, __U, __A, __B);
}

__m128 test_mm_fnmsub_round_ss(__m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_fnmsub_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[FMA]], i64 0
  return _mm_fnmsub_round_ss(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_fnmsub_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: test_mm_mask_fnmsub_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[A]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_mask_fnmsub_round_ss(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_fnmsub_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_maskz_fnmsub_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float 0.000000e+00
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_maskz_fnmsub_ss(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fnmsub_round_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C){
  // CHECK-LABEL: test_mm_maskz_fnmsub_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float 0.000000e+00
  // CHECK-NEXT: insertelement <4 x float> [[ORIGA]], float [[SEL]], i64 0
  return _mm_maskz_fnmsub_round_ss(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask3_fnmsub_ss(__m128 __W, __m128 __X, __m128 __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fnmsub_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x float> [[ORIGC:%.+]]
  // CHECK: [[A:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.fma.f32(float [[A]], float [[B]], float [[C]])
  // CHECK-NEXT: [[C2:%.+]] = extractelement <4 x float> [[ORIGC]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[C2]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGC]], float [[SEL]], i64 0
  return _mm_mask3_fnmsub_ss(__W, __X, __Y, __U);
}

__m128 test_mm_mask3_fnmsub_round_ss(__m128 __W, __m128 __X, __m128 __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fnmsub_round_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x float> [[ORIGC:%.+]]
  // CHECK: [[A:%.+]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <4 x float> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call float @llvm.x86.avx512.vfmadd.f32(float [[A]], float [[B]], float [[C]], i32 11)
  // CHECK-NEXT: [[C2:%.+]] = extractelement <4 x float> [[ORIGC]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, float [[FMA]], float [[C2]]
  // CHECK-NEXT: insertelement <4 x float> [[ORIGC]], float [[SEL]], i64 0
  return _mm_mask3_fnmsub_round_ss(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_fmadd_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_fmadd_sd
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[A]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_mask_fmadd_sd(__W, __U, __A, __B);
}

__m128d test_mm_fmadd_round_sd(__m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_fmadd_round_sd
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[FMA]], i64 0
  return _mm_fmadd_round_sd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_fmadd_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_fmadd_round_sd
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[A]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_mask_fmadd_round_sd(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_fmadd_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_maskz_fmadd_sd
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double 0.000000e+00
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_maskz_fmadd_sd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fmadd_round_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_maskz_fmadd_round_sd
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double 0.000000e+00
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_maskz_fmadd_round_sd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask3_fmadd_sd(__m128d __W, __m128d __X, __m128d __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fmadd_sd
  // CHECK: [[A:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[C]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGC]], double [[SEL]], i64 0
  return _mm_mask3_fmadd_sd(__W, __X, __Y, __U);
}

__m128d test_mm_mask3_fmadd_round_sd(__m128d __W, __m128d __X, __m128d __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fmadd_round_sd
  // CHECK: [[A:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[C]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGC]], double [[SEL]], i64 0
  return _mm_mask3_fmadd_round_sd(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_fmsub_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_fmsub_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[A]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_mask_fmsub_sd(__W, __U, __A, __B);
}

__m128d test_mm_fmsub_round_sd(__m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_fmsub_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[FMA]], i64 0
  return _mm_fmsub_round_sd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_fmsub_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_fmsub_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[A]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_mask_fmsub_round_sd(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_fmsub_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_maskz_fmsub_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double 0.000000e+00
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_maskz_fmsub_sd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fmsub_round_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_maskz_fmsub_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double 0.000000e+00
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_maskz_fmsub_round_sd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask3_fmsub_sd(__m128d __W, __m128d __X, __m128d __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fmsub_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> [[ORIGC:%.+]]
  // CHECK: [[A:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: [[C2:%.+]] = extractelement <2 x double> [[ORIGC]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[C2]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGC]], double [[SEL]], i64 0
  return _mm_mask3_fmsub_sd(__W, __X, __Y, __U);
}

__m128d test_mm_mask3_fmsub_round_sd(__m128d __W, __m128d __X, __m128d __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fmsub_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> [[ORIGC:%.+]]
  // CHECK: [[A:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: [[C2:%.+]] = extractelement <2 x double> [[ORIGC]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[C2]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGC]], double [[SEL]], i64 0
  return _mm_mask3_fmsub_round_sd(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_fnmadd_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_fnmadd_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[A]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_mask_fnmadd_sd(__W, __U, __A, __B);
}

__m128d test_mm_fnmadd_round_sd(__m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_fnmadd_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[FMA]], i64 0
  return _mm_fnmadd_round_sd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_fnmadd_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_fnmadd_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[A]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_mask_fnmadd_round_sd(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_fnmadd_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_maskz_fnmadd_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double 0.000000e+00
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_maskz_fnmadd_sd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fnmadd_round_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_maskz_fnmadd_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double 0.000000e+00
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_maskz_fnmadd_round_sd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask3_fnmadd_sd(__m128d __W, __m128d __X, __m128d __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fnmadd_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[C]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGC]], double [[SEL]], i64 0
  return _mm_mask3_fnmadd_sd(__W, __X, __Y, __U);
}

__m128d test_mm_mask3_fnmadd_round_sd(__m128d __W, __m128d __X, __m128d __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fnmadd_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[C]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGC]], double [[SEL]], i64 0
  return _mm_mask3_fnmadd_round_sd(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_fnmsub_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_fnmsub_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[A]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_mask_fnmsub_sd(__W, __U, __A, __B);
}

__m128d test_mm_fnmsub_round_sd(__m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_fnmsub_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[FMA]], i64 0
  return _mm_fnmsub_round_sd(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_fnmsub_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: test_mm_mask_fnmsub_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[A]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_mask_fnmsub_round_sd(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_fnmsub_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_maskz_fnmsub_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double 0.000000e+00
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_maskz_fnmsub_sd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fnmsub_round_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C){
  // CHECK-LABEL: test_mm_maskz_fnmsub_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[ORIGA:%.]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double 0.000000e+00
  // CHECK-NEXT: insertelement <2 x double> [[ORIGA]], double [[SEL]], i64 0
  return _mm_maskz_fnmsub_round_sd(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask3_fnmsub_sd(__m128d __W, __m128d __X, __m128d __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fnmsub_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <2 x double> [[ORIGC:%.+]]
  // CHECK: [[A:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
  // CHECK-NEXT: [[C2:%.+]] = extractelement <2 x double> [[ORIGC]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[C2]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGC]], double [[SEL]], i64 0
  return _mm_mask3_fnmsub_sd(__W, __X, __Y, __U);
}

__m128d test_mm_mask3_fnmsub_round_sd(__m128d __W, __m128d __X, __m128d __Y, __mmask8 __U){
  // CHECK-LABEL: test_mm_mask3_fnmsub_round_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <2 x double> [[ORIGC:%.+]]
  // CHECK: [[A:%.+]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <2 x double> [[NEG2]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call double @llvm.x86.avx512.vfmadd.f64(double [[A]], double [[B]], double [[C]], i32 11)
  // CHECK-NEXT: [[C2:%.+]] = extractelement <2 x double> [[ORIGC]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, double [[FMA]], double [[C2]]
  // CHECK-NEXT: insertelement <2 x double> [[ORIGC]], double [[SEL]], i64 0
  return _mm_mask3_fnmsub_round_sd(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_permutex_pd(__m512d __X) {
  // CHECK-LABEL: test_mm512_permutex_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  return _mm512_permutex_pd(__X, 0);
}

__m512d test_mm512_mask_permutex_pd(__m512d __W, __mmask8 __U, __m512d __X) {
  // CHECK-LABEL: test_mm512_mask_permutex_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_permutex_pd(__W, __U, __X, 0);
}

__m512d test_mm512_maskz_permutex_pd(__mmask8 __U, __m512d __X) {
  // CHECK-LABEL: test_mm512_maskz_permutex_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_permutex_pd(__U, __X, 0);
}

__m512i test_mm512_permutex_epi64(__m512i __X) {
  // CHECK-LABEL: test_mm512_permutex_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  return _mm512_permutex_epi64(__X, 0);
}

__m512i test_mm512_mask_permutex_epi64(__m512i __W, __mmask8 __M, __m512i __X) {
  // CHECK-LABEL: test_mm512_mask_permutex_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_permutex_epi64(__W, __M, __X, 0);
}

__m512i test_mm512_maskz_permutex_epi64(__mmask8 __M, __m512i __X) {
  // CHECK-LABEL: test_mm512_maskz_permutex_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_permutex_epi64(__M, __X, 0);
}

__m512d test_mm512_permutexvar_pd(__m512i __X, __m512d __Y) {
  // CHECK-LABEL: test_mm512_permutexvar_pd
  // CHECK: @llvm.x86.avx512.permvar.df.512
  return _mm512_permutexvar_pd(__X, __Y); 
}

__m512d test_mm512_mask_permutexvar_pd(__m512d __W, __mmask8 __U, __m512i __X, __m512d __Y) {
  // CHECK-LABEL: test_mm512_mask_permutexvar_pd
  // CHECK: @llvm.x86.avx512.permvar.df.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_permutexvar_pd(__W, __U, __X, __Y); 
}

__m512d test_mm512_maskz_permutexvar_pd(__mmask8 __U, __m512i __X, __m512d __Y) {
  // CHECK-LABEL: test_mm512_maskz_permutexvar_pd
  // CHECK: @llvm.x86.avx512.permvar.df.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_permutexvar_pd(__U, __X, __Y); 
}

__m512i test_mm512_maskz_permutexvar_epi64(__mmask8 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_maskz_permutexvar_epi64
  // CHECK: @llvm.x86.avx512.permvar.di.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_permutexvar_epi64(__M, __X, __Y); 
}

__m512i test_mm512_permutexvar_epi64(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_permutexvar_epi64
  // CHECK: @llvm.x86.avx512.permvar.di.512
  return _mm512_permutexvar_epi64(__X, __Y); 
}

__m512i test_mm512_mask_permutexvar_epi64(__m512i __W, __mmask8 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_permutexvar_epi64
  // CHECK: @llvm.x86.avx512.permvar.di.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_permutexvar_epi64(__W, __M, __X, __Y); 
}

__m512 test_mm512_permutexvar_ps(__m512i __X, __m512 __Y) {
  // CHECK-LABEL: test_mm512_permutexvar_ps
  // CHECK: @llvm.x86.avx512.permvar.sf.512
  return _mm512_permutexvar_ps(__X, __Y); 
}

__m512 test_mm512_mask_permutexvar_ps(__m512 __W, __mmask16 __U, __m512i __X, __m512 __Y) {
  // CHECK-LABEL: test_mm512_mask_permutexvar_ps
  // CHECK: @llvm.x86.avx512.permvar.sf.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_permutexvar_ps(__W, __U, __X, __Y); 
}

__m512 test_mm512_maskz_permutexvar_ps(__mmask16 __U, __m512i __X, __m512 __Y) {
  // CHECK-LABEL: test_mm512_maskz_permutexvar_ps
  // CHECK: @llvm.x86.avx512.permvar.sf.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_permutexvar_ps(__U, __X, __Y); 
}

__m512i test_mm512_maskz_permutexvar_epi32(__mmask16 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_maskz_permutexvar_epi32
  // CHECK: @llvm.x86.avx512.permvar.si.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_permutexvar_epi32(__M, __X, __Y); 
}

__m512i test_mm512_permutexvar_epi32(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_permutexvar_epi32
  // CHECK: @llvm.x86.avx512.permvar.si.512
  return _mm512_permutexvar_epi32(__X, __Y); 
}

__m512i test_mm512_mask_permutexvar_epi32(__m512i __W, __mmask16 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_permutexvar_epi32
  // CHECK: @llvm.x86.avx512.permvar.si.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_permutexvar_epi32(__W, __M, __X, __Y); 
}

__mmask16 test_mm512_kand(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_mm512_kand
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = and <16 x i1> [[LHS]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_mm512_kand(_mm512_cmpneq_epu32_mask(__A, __B),
                                                   _mm512_cmpneq_epu32_mask(__C, __D)),
                                                   __E, __F);
}

__mmask16 test_mm512_kandn(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_mm512_kandn
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[NOT:%.*]] = xor <16 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = and <16 x i1> [[NOT]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_mm512_kandn(_mm512_cmpneq_epu32_mask(__A, __B),
                                                    _mm512_cmpneq_epu32_mask(__C, __D)),
                                                    __E, __F);
}

__mmask16 test_mm512_kor(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_mm512_kor
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = or <16 x i1> [[LHS]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_mm512_kor(_mm512_cmpneq_epu32_mask(__A, __B),
                                                  _mm512_cmpneq_epu32_mask(__C, __D)),
                                                  __E, __F);
}

int test_mm512_kortestc(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_mm512_kortestc
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[OR:%.*]] = or <16 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <16 x i1> [[OR]] to i16
  // CHECK: [[CMP:%.*]] = icmp eq i16 [[CAST]], -1
  // CHECK: zext i1 [[CMP]] to i32
  return _mm512_kortestc(_mm512_cmpneq_epu32_mask(__A, __B),
                         _mm512_cmpneq_epu32_mask(__C, __D));
}

int test_mm512_kortestz(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_mm512_kortestz
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[OR:%.*]] = or <16 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <16 x i1> [[OR]] to i16
  // CHECK: [[CMP:%.*]] = icmp eq i16 [[CAST]], 0
  // CHECK: zext i1 [[CMP]] to i32
  return _mm512_kortestz(_mm512_cmpneq_epu32_mask(__A, __B),
                         _mm512_cmpneq_epu32_mask(__C, __D));
}

unsigned char test_kortestz_mask16_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_kortestz_mask16_u8
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[OR:%.*]] = or <16 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <16 x i1> [[OR]] to i16
  // CHECK: [[CMP:%.*]] = icmp eq i16 [[CAST]], 0
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  return _kortestz_mask16_u8(_mm512_cmpneq_epu32_mask(__A, __B),
                             _mm512_cmpneq_epu32_mask(__C, __D));
}

unsigned char test_kortestc_mask16_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_kortestc_mask16_u8
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[OR:%.*]] = or <16 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <16 x i1> [[OR]] to i16
  // CHECK: [[CMP:%.*]] = icmp eq i16 [[CAST]], -1
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  return _kortestc_mask16_u8(_mm512_cmpneq_epu32_mask(__A, __B),
                             _mm512_cmpneq_epu32_mask(__C, __D));
}

unsigned char test_kortest_mask16_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, unsigned char *CF) {
  // CHECK-LABEL: test_kortest_mask16_u8
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[OR:%.*]] = or <16 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <16 x i1> [[OR]] to i16
  // CHECK: [[CMP:%.*]] = icmp eq i16 [[CAST]], -1
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  // CHECK: [[LHS2:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS2:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[OR2:%.*]] = or <16 x i1> [[LHS2]], [[RHS2]]
  // CHECK: [[CAST2:%.*]] = bitcast <16 x i1> [[OR2]] to i16
  // CHECK: [[CMP2:%.*]] = icmp eq i16 [[CAST2]], 0
  // CHECK: [[ZEXT2:%.*]] = zext i1 [[CMP2]] to i32
  // CHECK: trunc i32 [[ZEXT2]] to i8
  return _kortest_mask16_u8(_mm512_cmpneq_epu32_mask(__A, __B),
                            _mm512_cmpneq_epu32_mask(__C, __D), CF);
}

__mmask16 test_mm512_kunpackb(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_mm512_kunpackb
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[LHS2:%.*]] = shufflevector <16 x i1> [[LHS]], <16 x i1> [[LHS]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: [[RHS2:%.*]] = shufflevector <16 x i1> [[RHS]], <16 x i1> [[RHS]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: [[CONCAT:%.*]] = shufflevector <8 x i1> [[RHS2]], <8 x i1> [[LHS2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: bitcast <16 x i1> [[CONCAT]] to i16
  return _mm512_mask_cmpneq_epu32_mask(_mm512_kunpackb(_mm512_cmpneq_epu32_mask(__A, __B),
                                                       _mm512_cmpneq_epu32_mask(__C, __D)),
                                                       __E, __F);
}

__mmask16 test_mm512_kxnor(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_mm512_kxnor
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[NOT:%.*]] = xor <16 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = xor <16 x i1> [[NOT]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_mm512_kxnor(_mm512_cmpneq_epu32_mask(__A, __B),
                                                    _mm512_cmpneq_epu32_mask(__C, __D)),
                                                    __E, __F);
}

__mmask16 test_mm512_kxor(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_mm512_kxor
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = xor <16 x i1> [[LHS]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_mm512_kxor(_mm512_cmpneq_epu32_mask(__A, __B),
                                                   _mm512_cmpneq_epu32_mask(__C, __D)),
                                                   __E, __F);
}

__mmask16 test_knot_mask16(__mmask16 a) {
  // CHECK-LABEL: test_knot_mask16
  // CHECK: [[IN:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[NOT:%.*]] = xor <16 x i1> [[IN]], splat (i1 true)
  // CHECK: bitcast <16 x i1> [[NOT]] to i16
  return _knot_mask16(a);
}

__mmask16 test_kand_mask16(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kand_mask16
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = and <16 x i1> [[LHS]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_kand_mask16(_mm512_cmpneq_epu32_mask(__A, __B),
                                                    _mm512_cmpneq_epu32_mask(__C, __D)),
                                                    __E, __F);
}

__mmask16 test_kandn_mask16(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kandn_mask16
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[NOT:%.*]] = xor <16 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = and <16 x i1> [[NOT]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_kandn_mask16(_mm512_cmpneq_epu32_mask(__A, __B),
                                                     _mm512_cmpneq_epu32_mask(__C, __D)),
                                                     __E, __F);
}

__mmask16 test_kor_mask16(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kor_mask16
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = or <16 x i1> [[LHS]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_kor_mask16(_mm512_cmpneq_epu32_mask(__A, __B),
                                                   _mm512_cmpneq_epu32_mask(__C, __D)),
                                                   __E, __F);
}

__mmask16 test_kxnor_mask16(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kxnor_mask16
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[NOT:%.*]] = xor <16 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = xor <16 x i1> [[NOT]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_kxnor_mask16(_mm512_cmpneq_epu32_mask(__A, __B),
                                                     _mm512_cmpneq_epu32_mask(__C, __D)),
                                                     __E, __F);
}

__mmask16 test_kxor_mask16(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kxor_mask16
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = xor <16 x i1> [[LHS]], [[RHS]]
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_kxor_mask16(_mm512_cmpneq_epu32_mask(__A, __B),
                                                    _mm512_cmpneq_epu32_mask(__C, __D)),
                                                    __E, __F);
}

__mmask16 test_kshiftli_mask16(__m512i A, __m512i B, __m512i C, __m512i D) {
  // CHECK-LABEL: test_kshiftli_mask16
  // CHECK: [[VAL:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = shufflevector <16 x i1> zeroinitializer, <16 x i1> [[VAL]], <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_kshiftli_mask16(_mm512_cmpneq_epu32_mask(A, B), 1), C, D);
}

__mmask16 test_kshiftri_mask16(__m512i A, __m512i B, __m512i C, __m512i D) {
  // CHECK-LABEL: test_kshiftri_mask16
  // CHECK: [[VAL:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = shufflevector <16 x i1> [[VAL]], <16 x i1> zeroinitializer, <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
  // CHECK: bitcast <16 x i1> {{.*}} to i16
  return _mm512_mask_cmpneq_epu32_mask(_kshiftri_mask16(_mm512_cmpneq_epu32_mask(A, B), 1), C, D);
}

unsigned int test_cvtmask16_u32(__m512i A, __m512i B) {
  // CHECK-LABEL: test_cvtmask16_u32
  // CHECK: bitcast <16 x i1> %{{.*}} to i16
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: zext i16 %{{.*}} to i32
  return _cvtmask16_u32(_mm512_cmpneq_epu32_mask(A, B));
}

__mmask16 test_cvtu32_mask16(__m512i A, __m512i B, unsigned int C) {
  // CHECK-LABEL: test_cvtu32_mask16
  // CHECK: trunc i32 %{{.*}} to i16
  // CHECK: bitcast i16 %{{.*}} to <16 x i1>
  return _mm512_mask_cmpneq_epu32_mask(_cvtu32_mask16(C), A, B);
}

__mmask16 test_load_mask16(__mmask16 *A, __m512i B, __m512i C) {
  // CHECK-LABEL: test_load_mask16
  // CHECK: [[LOAD:%.*]] = load i16, ptr %{{.*}}{{$}}
  // CHECK: bitcast i16 [[LOAD]] to <16 x i1>
  return _mm512_mask_cmpneq_epu32_mask(_load_mask16(A), B, C);
}

void test_store_mask16(__mmask16 *A, __m512i B, __m512i C) {
  // CHECK-LABEL: test_store_mask16
  // CHECK: bitcast <16 x i1> %{{.*}} to i16
  // CHECK: store i16 %{{.*}}, ptr %{{.*}}
  _store_mask16(A, _mm512_cmpneq_epu32_mask(B, C));
}

void test_mm512_stream_si512(__m512i * __P, __m512i __A) {
  // CHECK-LABEL: test_mm512_stream_si512
  // CHECK: store <8 x i64> %{{.*}}, ptr %{{.*}}, align 64, !nontemporal [[NONTEMPORAL:![0-9]+]]
  _mm512_stream_si512(__P, __A); 
}

void test_mm512_stream_si512_2(void * __P, __m512i __A) {
  // CHECK-LABEL: test_mm512_stream_si512
  // CHECK: store <8 x i64> %{{.*}}, ptr %{{.*}}, align 64, !nontemporal [[NONTEMPORAL]]
  _mm512_stream_si512(__P, __A); 
}

__m512i test_mm512_stream_load_si512(void *__P) {
  // CHECK-LABEL: test_mm512_stream_load_si512
  // CHECK: load <8 x i64>, ptr %{{.*}}, align 64, !nontemporal [[NONTEMPORAL]]{{$}}
  return _mm512_stream_load_si512(__P); 
}

__m512i test_mm512_stream_load_si512_const(void const *__P) {
  // CHECK-LABEL: test_mm512_stream_load_si512_const
  // CHECK: load <8 x i64>, ptr %{{.*}}, align 64, !nontemporal [[NONTEMPORAL]]{{$}}
  return _mm512_stream_load_si512(__P); 
}

void test_mm512_stream_pd(double *__P, __m512d __A) {
  // CHECK-LABEL: test_mm512_stream_pd
  // CHECK: store <8 x double> %{{.*}}, ptr %{{.*}}, align 64, !nontemporal [[NONTEMPORAL]]
  return _mm512_stream_pd(__P, __A); 
}

void test_mm512_stream_pd_2(void *__P, __m512d __A) {
  // CHECK-LABEL: test_mm512_stream_pd
  // CHECK: store <8 x double> %{{.*}}, ptr %{{.*}}, align 64, !nontemporal [[NONTEMPORAL]]
  return _mm512_stream_pd(__P, __A); 
}

void test_mm512_stream_ps(float *__P, __m512 __A) {
  // CHECK-LABEL: test_mm512_stream_ps
  // CHECK: store <16 x float> %{{.*}}, ptr %{{.*}}, align 64, !nontemporal [[NONTEMPORAL]]
  _mm512_stream_ps(__P, __A); 
}

void test_mm512_stream_ps_2(void *__P, __m512 __A) {
  // CHECK-LABEL: test_mm512_stream_ps
  // CHECK: store <16 x float> %{{.*}}, ptr %{{.*}}, align 64, !nontemporal [[NONTEMPORAL]]
  _mm512_stream_ps(__P, __A); 
}
__m512d test_mm512_mask_compress_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_mask_compress_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_compress_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_maskz_compress_pd(__U, __A); 
}

__m512i test_mm512_mask_compress_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_mask_compress_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_compress_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_maskz_compress_epi64(__U, __A); 
}

__m512 test_mm512_mask_compress_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_mask_compress_ps(__W, __U, __A); 
}

__m512 test_mm512_maskz_compress_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_maskz_compress_ps(__U, __A); 
}

__m512i test_mm512_mask_compress_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_mask_compress_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_compress_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_maskz_compress_epi32(__U, __A); 
}

__mmask8 test_mm_cmp_round_ss_mask(__m128 __X, __m128 __Y) {
  // CHECK-LABEL: test_mm_cmp_round_ss_mask
  // CHECK: @llvm.x86.avx512.mask.cmp
  return _mm_cmp_round_ss_mask(__X, __Y, _CMP_NLT_US, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm_mask_cmp_round_ss_mask(__mmask8 __M, __m128 __X, __m128 __Y) {
  // CHECK-LABEL: test_mm_mask_cmp_round_ss_mask
  // CHECK: @llvm.x86.avx512.mask.cmp
  return _mm_mask_cmp_round_ss_mask(__M, __X, __Y, _CMP_NLT_US, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm_cmp_ss_mask(__m128 __X, __m128 __Y) {
  // CHECK-LABEL: test_mm_cmp_ss_mask
  // CHECK: @llvm.x86.avx512.mask.cmp
  return _mm_cmp_ss_mask(__X, __Y, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_ss_mask(__mmask8 __M, __m128 __X, __m128 __Y) {
  // CHECK-LABEL: test_mm_mask_cmp_ss_mask
  // CHECK: @llvm.x86.avx512.mask.cmp
  return _mm_mask_cmp_ss_mask(__M, __X, __Y, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_round_sd_mask(__m128d __X, __m128d __Y) {
  // CHECK-LABEL: test_mm_cmp_round_sd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp
  return _mm_cmp_round_sd_mask(__X, __Y, _CMP_NLT_US, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm_mask_cmp_round_sd_mask(__mmask8 __M, __m128d __X, __m128d __Y) {
  // CHECK-LABEL: test_mm_mask_cmp_round_sd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp
  return _mm_mask_cmp_round_sd_mask(__M, __X, __Y, _CMP_NLT_US, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm_cmp_sd_mask(__m128d __X, __m128d __Y) {
  // CHECK-LABEL: test_mm_cmp_sd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp
  return _mm_cmp_sd_mask(__X, __Y, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_sd_mask(__mmask8 __M, __m128d __X, __m128d __Y) {
  // CHECK-LABEL: test_mm_mask_cmp_sd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp
  return _mm_mask_cmp_sd_mask(__M, __X, __Y, _CMP_NLT_US);
}

__m512 test_mm512_movehdup_ps(__m512 __A) {
  // CHECK-LABEL: test_mm512_movehdup_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7, i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15>
  return _mm512_movehdup_ps(__A);
}

__m512 test_mm512_mask_movehdup_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_movehdup_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7, i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_movehdup_ps(__W, __U, __A);
}

__m512 test_mm512_maskz_movehdup_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_movehdup_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7, i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_movehdup_ps(__U, __A);
}

__m512 test_mm512_moveldup_ps(__m512 __A) {
  // CHECK-LABEL: test_mm512_moveldup_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6, i32 8, i32 8, i32 10, i32 10, i32 12, i32 12, i32 14, i32 14>
  return _mm512_moveldup_ps(__A);
}

__m512 test_mm512_mask_moveldup_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_moveldup_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6, i32 8, i32 8, i32 10, i32 10, i32 12, i32 12, i32 14, i32 14>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_moveldup_ps(__W, __U, __A);
}

__m512 test_mm512_maskz_moveldup_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_moveldup_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6, i32 8, i32 8, i32 10, i32 10, i32 12, i32 12, i32 14, i32 14>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_moveldup_ps(__U, __A);
}

__m512i test_mm512_shuffle_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_shuffle_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <16 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4, i32 9, i32 8, i32 8, i32 8, i32 13, i32 12, i32 12, i32 12>
  return _mm512_shuffle_epi32(__A, 1); 
}

__m512i test_mm512_mask_shuffle_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_shuffle_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <16 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4, i32 9, i32 8, i32 8, i32 8, i32 13, i32 12, i32 12, i32 12>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shuffle_epi32(__W, __U, __A, 1); 
}

__m512i test_mm512_maskz_shuffle_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_shuffle_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <16 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4, i32 9, i32 8, i32 8, i32 8, i32 13, i32 12, i32 12, i32 12>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shuffle_epi32(__U, __A, 1); 
}

__m512d test_mm512_mask_expand_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_mask_expand_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_expand_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_maskz_expand_pd(__U, __A); 
}

__m512i test_mm512_mask_expand_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_mask_expand_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_expand_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_maskz_expand_epi64(__U, __A); 
}
__m512i test_mm512_mask_expandloadu_epi64(__m512i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_mask_expandloadu_epi64
  // CHECK: @llvm.masked.expandload.v8i64(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_mask_expandloadu_epi64(__W, __U, __P); 
}

__m512i test_mm512_maskz_expandloadu_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_maskz_expandloadu_epi64
  // CHECK: @llvm.masked.expandload.v8i64(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_maskz_expandloadu_epi64(__U, __P); 
}

__m512d test_mm512_mask_expandloadu_pd(__m512d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_mask_expandloadu_pd
  // CHECK: @llvm.masked.expandload.v8f64(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_mask_expandloadu_pd(__W, __U, __P); 
}

__m512d test_mm512_maskz_expandloadu_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_maskz_expandloadu_pd
  // CHECK: @llvm.masked.expandload.v8f64(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_maskz_expandloadu_pd(__U, __P); 
}

__m512i test_mm512_mask_expandloadu_epi32(__m512i __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_mask_expandloadu_epi32
  // CHECK: @llvm.masked.expandload.v16i32(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_mask_expandloadu_epi32(__W, __U, __P); 
}

__m512i test_mm512_maskz_expandloadu_epi32(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_maskz_expandloadu_epi32
  // CHECK: @llvm.masked.expandload.v16i32(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_maskz_expandloadu_epi32(__U, __P); 
}

__m512 test_mm512_mask_expandloadu_ps(__m512 __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_mask_expandloadu_ps
  // CHECK: @llvm.masked.expandload.v16f32(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_mask_expandloadu_ps(__W, __U, __P); 
}

__m512 test_mm512_maskz_expandloadu_ps(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm512_maskz_expandloadu_ps
  // CHECK: @llvm.masked.expandload.v16f32(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_maskz_expandloadu_ps(__U, __P); 
}

__m512 test_mm512_mask_expand_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_mask_expand_ps(__W, __U, __A); 
}

__m512 test_mm512_maskz_expand_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_maskz_expand_ps(__U, __A); 
}

__m512i test_mm512_mask_expand_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_mask_expand_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_expand_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_maskz_expand_epi32(__U, __A); 
}
__m512d test_mm512_cvt_roundps_pd(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvt_roundps_pd
  // CHECK: @llvm.x86.avx512.mask.cvtps2pd.512
  return _mm512_cvt_roundps_pd(__A, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_cvt_roundps_pd(__m512d __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundps_pd
  // CHECK: @llvm.x86.avx512.mask.cvtps2pd.512
  return _mm512_mask_cvt_roundps_pd(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_cvt_roundps_pd(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundps_pd
  // CHECK: @llvm.x86.avx512.mask.cvtps2pd.512
  return _mm512_maskz_cvt_roundps_pd(__U, __A, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_cvtps_pd(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvtps_pd
  // CHECK: fpext <8 x float> %{{.*}} to <8 x double>
  return _mm512_cvtps_pd(__A); 
}

__m512d test_mm512_cvtpslo_pd(__m512 __A) {
  // CHECK-LABEL: test_mm512_cvtpslo_pd
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: fpext <8 x float> %{{.*}} to <8 x double>
  return _mm512_cvtpslo_pd(__A);
}

__m512d test_mm512_mask_cvtps_pd(__m512d __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvtps_pd
  // CHECK: fpext <8 x float> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_cvtps_pd(__W, __U, __A); 
}

__m512d test_mm512_mask_cvtpslo_pd(__m512d __W, __mmask8 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_cvtpslo_pd
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: fpext <8 x float> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_cvtpslo_pd(__W, __U, __A);
}

__m512d test_mm512_maskz_cvtps_pd(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtps_pd
  // CHECK: fpext <8 x float> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_cvtps_pd(__U, __A); 
}
__m512d test_mm512_mask_mov_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_mov_pd
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_mov_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_mov_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_mov_pd
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_mov_pd(__U, __A); 
}

__m512 test_mm512_mask_mov_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_mov_ps
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_mov_ps(__W, __U, __A); 
}

__m512 test_mm512_maskz_mov_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_mov_ps
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_mov_ps(__U, __A); 
}

void test_mm512_mask_compressstoreu_pd(void *__P, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_compressstoreu_pd
  // CHECK: @llvm.masked.compressstore.v8f64(<8 x double> %{{.*}}, ptr %{{.*}}, <8 x i1> %{{.*}})
  return _mm512_mask_compressstoreu_pd(__P, __U, __A); 
}

void test_mm512_mask_compressstoreu_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_compressstoreu_epi64
  // CHECK: @llvm.masked.compressstore.v8i64(<8 x i64> %{{.*}}, ptr %{{.*}}, <8 x i1> %{{.*}})
  return _mm512_mask_compressstoreu_epi64(__P, __U, __A); 
}

void test_mm512_mask_compressstoreu_ps(void *__P, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_compressstoreu_ps
  // CHECK: @llvm.masked.compressstore.v16f32(<16 x float> %{{.*}}, ptr %{{.*}}, <16 x i1> %{{.*}})
  return _mm512_mask_compressstoreu_ps(__P, __U, __A); 
}

void test_mm512_mask_compressstoreu_epi32(void *__P, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_compressstoreu_epi32
  // CHECK: @llvm.masked.compressstore.v16i32(<16 x i32> %{{.*}}, ptr %{{.*}}, <16 x i1> %{{.*}})
  return _mm512_mask_compressstoreu_epi32(__P, __U, __A); 
}

__m256i test_mm512_cvtt_roundpd_epu32(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvtt_roundpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.512
  return _mm512_cvtt_roundpd_epu32(__A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvtt_roundpd_epu32(__m256i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.512
  return _mm512_mask_cvtt_roundpd_epu32(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvtt_roundpd_epu32(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.512
  return _mm512_maskz_cvtt_roundpd_epu32(__U, __A, _MM_FROUND_NO_EXC);
}

__m256i test_mm512_cvttpd_epu32(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.512
  return _mm512_cvttpd_epu32(__A); 
}

__m256i test_mm512_mask_cvttpd_epu32(__m256i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.512
  return _mm512_mask_cvttpd_epu32(__W, __U, __A); 
}

__m256i test_mm512_maskz_cvttpd_epu32(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.512
  return _mm512_maskz_cvttpd_epu32(__U, __A); 
}

__m512 test_mm512_castpd_ps (__m512d __A)
{
  // CHECK-LABEL: test_mm512_castpd_ps 
  // CHECK: bitcast <8 x double> %{{.}} to <16 x float>
  return _mm512_castpd_ps (__A);
}

__m512d test_mm512_castps_pd (__m512 __A)
{
  // CHECK-LABEL: test_mm512_castps_pd 
  // CHECK: bitcast <16 x float> %{{.}} to <8 x double>
  return _mm512_castps_pd (__A);
}

__m512i test_mm512_castpd_si512 (__m512d __A)
{
  // CHECK-LABEL: test_mm512_castpd_si512 
  // CHECK: bitcast <8 x double> %{{.}} to <8 x i64>
  return _mm512_castpd_si512 (__A);
}

__m512 test_mm512_castps128_ps512(__m128 __A) {
  // CHECK-LABEL: test_mm512_castps128_ps512
  // CHECK: [[B:%.*]] = freeze <8 x float> poison
  // CHECK: store <8 x float> [[B]], ptr [[BA:%.*]]
  // CHECK: [[A:%.*]] = freeze <4 x float> poison 
  // CHECK: [[SV:%.*]] = shufflevector <4 x float> %{{.*}}, <4 x float> [[A]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: [[C:%.*]] = load <8 x float>, ptr [[BA]]
  // CHECK: shufflevector <8 x float> [[SV]], <8 x float> [[C]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_castps128_ps512(__A); 
}

__m512d test_mm512_castpd128_pd512(__m128d __A) {
  // CHECK-LABEL: test_mm512_castpd128_pd512
  // CHECK: [[B:%.*]] = freeze <4 x double> poison
  // CHECK: store <4 x double> [[B]], ptr [[BA:%.*]]
  // CHECK: [[A:%.*]] = freeze <2 x double> poison 
  // CHECK: [[SV:%.*]] = shufflevector <2 x double> %{{.*}}, <2 x double> [[A]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: [[C:%.*]] = load <4 x double>, ptr [[BA]]
  // CHECK: shufflevector <4 x double> [[SV]], <4 x double> [[C]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_castpd128_pd512(__A); 
}

__m512i test_mm512_set1_epi8(char d)
{
  // CHECK-LABEL: test_mm512_set1_epi8
  // CHECK: insertelement <64 x i8> {{.*}}, i32 0
  // CHECK: insertelement <64 x i8> {{.*}}, i32 1
  // CHECK: insertelement <64 x i8> {{.*}}, i32 2
  // CHECK: insertelement <64 x i8> {{.*}}, i32 3
  // CHECK: insertelement <64 x i8> {{.*}}, i32 4
  // CHECK: insertelement <64 x i8> {{.*}}, i32 5
  // CHECK: insertelement <64 x i8> {{.*}}, i32 6
  // CHECK: insertelement <64 x i8> {{.*}}, i32 7
  // CHECK: insertelement <64 x i8> {{.*}}, i32 63
  return _mm512_set1_epi8(d);
}

__m512i test_mm512_set1_epi16(short d)
{
  // CHECK-LABEL: test_mm512_set1_epi16
  // CHECK: insertelement <32 x i16> {{.*}}, i32 0
  // CHECK: insertelement <32 x i16> {{.*}}, i32 1
  // CHECK: insertelement <32 x i16> {{.*}}, i32 2
  // CHECK: insertelement <32 x i16> {{.*}}, i32 3
  // CHECK: insertelement <32 x i16> {{.*}}, i32 4
  // CHECK: insertelement <32 x i16> {{.*}}, i32 5
  // CHECK: insertelement <32 x i16> {{.*}}, i32 6
  // CHECK: insertelement <32 x i16> {{.*}}, i32 7
  // CHECK: insertelement <32 x i16> {{.*}}, i32 31
  return _mm512_set1_epi16(d);
}

__m512i test_mm512_set4_epi32 (int __A, int __B, int __C, int __D)
{
  // CHECK-LABEL: test_mm512_set4_epi32 
  // CHECK: insertelement <16 x i32> {{.*}}, i32 15
  return _mm512_set4_epi32 (__A,__B,__C,__D);
}

__m512i test_mm512_set4_epi64 (long long __A, long long __B, long long __C, long long __D)
{
  // CHECK-LABEL: test_mm512_set4_epi64 
  // CHECK: insertelement <8 x i64> {{.*}}, i32 7
  return _mm512_set4_epi64 (__A,__B,__C,__D);
}

__m512d test_mm512_set4_pd (double __A, double __B, double __C, double __D)
{
  // CHECK-LABEL: test_mm512_set4_pd 
  // CHECK: insertelement <8 x double> {{.*}}, i32 7
  return _mm512_set4_pd (__A,__B,__C,__D);
}

__m512 test_mm512_set4_ps (float __A, float __B, float __C, float __D)
{
  // CHECK-LABEL: test_mm512_set4_ps 
  // CHECK: insertelement <16 x float> {{.*}}, i32 15
  return _mm512_set4_ps (__A,__B,__C,__D);
}

__m512i test_mm512_setr4_epi32(int e0, int e1, int e2, int e3)
{
  // CHECK-LABEL: test_mm512_setr4_epi32
  // CHECK: insertelement <16 x i32> {{.*}}, i32 15
  return _mm512_setr4_epi32(e0, e1, e2, e3);
}

 __m512i test_mm512_setr4_epi64(long long e0, long long e1, long long e2, long long e3)
{
  // CHECK-LABEL: test_mm512_setr4_epi64
  // CHECK: insertelement <8 x i64> {{.*}}, i32 7
  return _mm512_setr4_epi64(e0, e1, e2, e3);
}

__m512d test_mm512_setr4_pd(double e0, double e1, double e2, double e3)
{
  // CHECK-LABEL: test_mm512_setr4_pd
  // CHECK: insertelement <8 x double> {{.*}}, i32 7
  return _mm512_setr4_pd(e0,e1,e2,e3);
}

 __m512 test_mm512_setr4_ps(float e0, float e1, float e2, float e3)
{
  // CHECK-LABEL: test_mm512_setr4_ps
  // CHECK: insertelement <16 x float> {{.*}}, i32 15
  return _mm512_setr4_ps(e0,e1,e2,e3);
}

__m512d test_mm512_castpd256_pd512(__m256d a)
{
  // CHECK-LABEL: test_mm512_castpd256_pd512
  // CHECK: [[A:%.*]] = freeze <4 x double> poison 
  // CHECK: shufflevector <4 x double> %{{.}}, <4 x double> [[A]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_castpd256_pd512(a);
}

__m256d test_mm512_castpd512_pd256 (__m512d __A)
{
  // CHECK-LABEL: test_mm512_castpd512_pd256 
  // CHECK: shufflevector <8 x double> %{{.}}, <8 x double> %{{.}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm512_castpd512_pd256 (__A);
}

__m256 test_mm512_castps512_ps256 (__m512 __A)
{
  // CHECK-LABEL: test_mm512_castps512_ps256 
  // CHECK: shufflevector <16 x float> %{{.}}, <16 x float> %{{.}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_castps512_ps256 (__A);
}

__m512i test_mm512_castps_si512 (__m512 __A)
{
  // CHECK-LABEL: test_mm512_castps_si512 
  // CHECK: bitcast <16 x float> %{{.}} to <8 x i64>
  return _mm512_castps_si512 (__A);
}
__m512i test_mm512_castsi128_si512(__m128i __A) {
  // CHECK-LABEL: test_mm512_castsi128_si512
  // CHECK: [[B:%.*]] = freeze <4 x i64> poison
  // CHECK: store <4 x i64> [[B]], ptr [[BA:%.*]]
  // CHECK: [[A:%.*]] = freeze <2 x i64> poison 
  // CHECK: [[SV:%.*]] = shufflevector <2 x i64> %{{.*}}, <2 x i64> [[A]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: [[C:%.*]] = load <4 x i64>, ptr [[BA]]
  // CHECK: shufflevector <4 x i64> [[SV]], <4 x i64> [[C]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_castsi128_si512(__A); 
}

__m512i test_mm512_castsi256_si512(__m256i __A) {
  // CHECK-LABEL: test_mm512_castsi256_si512
  // CHECK: [[A:%.*]] = freeze <4 x i64> poison 
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> [[A]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_castsi256_si512(__A); 
}

__m512 test_mm512_castsi512_ps (__m512i __A)
{
  // CHECK-LABEL: test_mm512_castsi512_ps 
  // CHECK: bitcast <8 x i64> %{{.}} to <16 x float>
  return _mm512_castsi512_ps (__A);
}

__m512d test_mm512_castsi512_pd (__m512i __A)
{
  // CHECK-LABEL: test_mm512_castsi512_pd 
  // CHECK: bitcast <8 x i64> %{{.}} to <8 x double>
  return _mm512_castsi512_pd (__A);
}

__m128i test_mm512_castsi512_si128 (__m512i __A)
{
  // CHECK-LABEL: test_mm512_castsi512_si128 
  // CHECK: shufflevector <8 x i64> %{{.}}, <8 x i64> %{{.}}, <2 x i32> <i32 0, i32 1>
  return _mm512_castsi512_si128 (__A);
}

__m256i test_mm512_castsi512_si256 (__m512i __A)
{
  // CHECK-LABEL: test_mm512_castsi512_si256 
  // CHECK: shufflevector <8 x i64> %{{.}}, <8 x i64> %{{.}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm512_castsi512_si256 (__A);
}

__m128 test_mm_cvt_roundsd_ss(__m128 __A, __m128d __B) {
  // CHECK-LABEL: test_mm_cvt_roundsd_ss
  // CHECK: @llvm.x86.avx512.mask.cvtsd2ss.round
  return _mm_cvt_roundsd_ss(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_cvt_roundsd_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_cvt_roundsd_ss
  // CHECK: @llvm.x86.avx512.mask.cvtsd2ss.round
  return _mm_mask_cvt_roundsd_ss(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_cvt_roundsd_ss(__mmask8 __U, __m128 __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_cvt_roundsd_ss
  // CHECK: @llvm.x86.avx512.mask.cvtsd2ss.round
  return _mm_maskz_cvt_roundsd_ss(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

#ifdef __x86_64__
__m128d test_mm_cvt_roundi64_sd(__m128d __A, long long __B) {
  // CHECK-LABEL: test_mm_cvt_roundi64_sd
  // CHECK: @llvm.x86.avx512.cvtsi2sd64
  return _mm_cvt_roundi64_sd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_cvt_roundsi64_sd(__m128d __A, long long __B) {
  // CHECK-LABEL: test_mm_cvt_roundsi64_sd
  // CHECK: @llvm.x86.avx512.cvtsi2sd64
  return _mm_cvt_roundsi64_sd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
#endif

__m128 test_mm_cvt_roundsi32_ss(__m128 __A, int __B) {
  // CHECK-LABEL: test_mm_cvt_roundsi32_ss
  // CHECK: @llvm.x86.avx512.cvtsi2ss32
  return _mm_cvt_roundsi32_ss(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_cvt_roundi32_ss(__m128 __A, int __B) {
  // CHECK-LABEL: test_mm_cvt_roundi32_ss
  // CHECK: @llvm.x86.avx512.cvtsi2ss32
  return _mm_cvt_roundi32_ss(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

#ifdef __x86_64__
__m128 test_mm_cvt_roundsi64_ss(__m128 __A, long long __B) {
  // CHECK-LABEL: test_mm_cvt_roundsi64_ss
  // CHECK: @llvm.x86.avx512.cvtsi2ss64
  return _mm_cvt_roundsi64_ss(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_cvt_roundi64_ss(__m128 __A, long long __B) {
  // CHECK-LABEL: test_mm_cvt_roundi64_ss
  // CHECK: @llvm.x86.avx512.cvtsi2ss64
  return _mm_cvt_roundi64_ss(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
#endif

__m128d test_mm_cvt_roundss_sd(__m128d __A, __m128 __B) {
  // CHECK-LABEL: test_mm_cvt_roundss_sd
  // CHECK: @llvm.x86.avx512.mask.cvtss2sd.round
  return _mm_cvt_roundss_sd(__A, __B, _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_cvt_roundss_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_cvt_roundss_sd
  // CHECK: @llvm.x86.avx512.mask.cvtss2sd.round
  return _mm_mask_cvt_roundss_sd(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_cvt_roundss_sd( __mmask8 __U, __m128d __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_cvt_roundss_sd
  // CHECK: @llvm.x86.avx512.mask.cvtss2sd.round
  return _mm_maskz_cvt_roundss_sd( __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m128d test_mm_cvtu32_sd(__m128d __A, unsigned __B) {
  // CHECK-LABEL: test_mm_cvtu32_sd
  // CHECK: uitofp i32 %{{.*}} to double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_cvtu32_sd(__A, __B); 
}

#ifdef __x86_64__
__m128d test_mm_cvt_roundu64_sd(__m128d __A, unsigned long long __B) {
  // CHECK-LABEL: test_mm_cvt_roundu64_sd
  // CHECK: @llvm.x86.avx512.cvtusi642sd
  return _mm_cvt_roundu64_sd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128d test_mm_cvtu64_sd(__m128d __A, unsigned long long __B) {
  // CHECK-LABEL: test_mm_cvtu64_sd
  // CHECK: uitofp i64 %{{.*}} to double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_cvtu64_sd(__A, __B); 
}
#endif

__m128 test_mm_cvt_roundu32_ss(__m128 __A, unsigned __B) {
  // CHECK-LABEL: test_mm_cvt_roundu32_ss
  // CHECK: @llvm.x86.avx512.cvtusi2ss
  return _mm_cvt_roundu32_ss(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_cvtu32_ss(__m128 __A, unsigned __B) {
  // CHECK-LABEL: test_mm_cvtu32_ss
  // CHECK: uitofp i32 %{{.*}} to float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_cvtu32_ss(__A, __B); 
}

#ifdef __x86_64__
__m128 test_mm_cvt_roundu64_ss(__m128 __A, unsigned long long __B) {
  // CHECK-LABEL: test_mm_cvt_roundu64_ss
  // CHECK: @llvm.x86.avx512.cvtusi642ss
    return _mm_cvt_roundu64_ss(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128 test_mm_cvtu64_ss(__m128 __A, unsigned long long __B) {
  // CHECK-LABEL: test_mm_cvtu64_ss
  // CHECK: uitofp i64 %{{.*}} to float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_cvtu64_ss(__A, __B); 
}
#endif

__m512i test_mm512_mask_cvttps_epu32 (__m512i __W, __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_cvttps_epu32 
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.512
  return _mm512_mask_cvttps_epu32 (__W,__U,__A);
}

__m512i test_mm512_maskz_cvttps_epu32 (__mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvttps_epu32 
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.512
  return _mm512_maskz_cvttps_epu32 (__U,__A);
}

__m512 test_mm512_cvtepu32_ps (__m512i __A)
{
  // CHECK-LABEL: test_mm512_cvtepu32_ps 
  // CHECK: uitofp <16 x i32> %{{.*}} to <16 x float>
  return _mm512_cvtepu32_ps (__A);
}

__m512 test_mm512_mask_cvtepu32_ps (__m512 __W, __mmask16 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtepu32_ps 
  // CHECK: uitofp <16 x i32> %{{.*}} to <16 x float>
  // CHECK: select <16 x i1> {{.*}}, <16 x float> {{.*}}, <16 x float> {{.*}}
  return _mm512_mask_cvtepu32_ps (__W,__U,__A);
}

__m512 test_mm512_maskz_cvtepu32_ps (__mmask16 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtepu32_ps 
  // CHECK: uitofp <16 x i32> %{{.*}} to <16 x float>
  // CHECK: select <16 x i1> {{.*}}, <16 x float> {{.*}}, <16 x float> {{.*}}
  return _mm512_maskz_cvtepu32_ps (__U,__A);
}

__m512d test_mm512_cvtepi32_pd (__m256i __A)
{
  // CHECK-LABEL: test_mm512_cvtepi32_pd
  // CHECK: sitofp <8 x i32> %{{.*}} to <8 x double>
  return _mm512_cvtepi32_pd (__A);
}

__m512d test_mm512_mask_cvtepi32_pd (__m512d __W, __mmask8 __U, __m256i __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtepi32_pd
  // CHECK: sitofp <8 x i32> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> {{.*}}, <8 x double> {{.*}}, <8 x double> {{.*}}
  return _mm512_mask_cvtepi32_pd (__W,__U,__A);
}

__m512d test_mm512_maskz_cvtepi32_pd (__mmask8 __U, __m256i __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtepi32_pd
  // CHECK: sitofp <8 x i32> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> {{.*}}, <8 x double> {{.*}}, <8 x double> {{.*}}
  return _mm512_maskz_cvtepi32_pd (__U,__A);
}

__m512d test_mm512_cvtepi32lo_pd (__m512i __A)
{
  // CHECK-LABEL: test_mm512_cvtepi32lo_pd
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sitofp <8 x i32> %{{.*}} to <8 x double>
  return _mm512_cvtepi32lo_pd (__A);
}

__m512d test_mm512_mask_cvtepi32lo_pd (__m512d __W, __mmask8 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtepi32lo_pd
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sitofp <8 x i32> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> {{.*}}, <8 x double> {{.*}}, <8 x double> {{.*}}
  return _mm512_mask_cvtepi32lo_pd (__W, __U, __A);
}

__m512 test_mm512_cvtepi32_ps (__m512i __A)
{
  // CHECK-LABEL: test_mm512_cvtepi32_ps 
  // CHECK: sitofp <16 x i32> %{{.*}} to <16 x float>
  return _mm512_cvtepi32_ps (__A);
}

__m512 test_mm512_mask_cvtepi32_ps (__m512 __W, __mmask16 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtepi32_ps 
  // CHECK: sitofp <16 x i32> %{{.*}} to <16 x float>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_cvtepi32_ps (__W,__U,__A);
}

__m512 test_mm512_maskz_cvtepi32_ps (__mmask16 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtepi32_ps 
  // CHECK: sitofp <16 x i32> %{{.*}} to <16 x float>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_cvtepi32_ps (__U,__A);
}

__m512d test_mm512_cvtepu32_pd(__m256i __A)
{
  // CHECK-LABEL: test_mm512_cvtepu32_pd
  // CHECK: uitofp <8 x i32> %{{.*}} to <8 x double>
  return _mm512_cvtepu32_pd(__A);
}

__m512d test_mm512_mask_cvtepu32_pd (__m512d __W, __mmask8 __U, __m256i __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtepu32_pd
  // CHECK: uitofp <8 x i32> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> {{.*}}, <8 x double> {{.*}}, <8 x double> {{.*}}
  return _mm512_mask_cvtepu32_pd (__W,__U,__A);
}

__m512d test_mm512_maskz_cvtepu32_pd (__mmask8 __U, __m256i __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtepu32_pd
  // CHECK: uitofp <8 x i32> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> {{.*}}, <8 x double> {{.*}}, <8 x double> {{.*}}
  return _mm512_maskz_cvtepu32_pd (__U,__A);
}

__m512d test_mm512_cvtepu32lo_pd (__m512i __A)
{
  // CHECK-LABEL: test_mm512_cvtepu32lo_pd
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: uitofp <8 x i32> %{{.*}} to <8 x double>
  return _mm512_cvtepu32lo_pd (__A);
}

__m512d test_mm512_mask_cvtepu32lo_pd (__m512d __W, __mmask8 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtepu32lo_pd
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: uitofp <8 x i32> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> {{.*}}, <8 x double> {{.*}}, <8 x double> {{.*}}
  return _mm512_mask_cvtepu32lo_pd (__W, __U, __A);
}

__m256 test_mm512_cvtpd_ps (__m512d __A)
{
  // CHECK-LABEL: test_mm512_cvtpd_ps 
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.512
  return _mm512_cvtpd_ps (__A);
}

__m256 test_mm512_mask_cvtpd_ps (__m256 __W, __mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtpd_ps 
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.512
  return _mm512_mask_cvtpd_ps (__W,__U,__A);
}

__m512 test_mm512_cvtpd_pslo(__m512d __A)
{
  // CHECK-LABEL: test_mm512_cvtpd_pslo
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.512
  // CHECK: zeroinitializer
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_cvtpd_pslo(__A);
}

__m512 test_mm512_mask_cvtpd_pslo(__m512 __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvtpd_pslo
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.512
  // CHECK: zeroinitializer
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_mask_cvtpd_pslo(__W, __U, __A);
}

__m256 test_mm512_maskz_cvtpd_ps (__mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtpd_ps 
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.512
  return _mm512_maskz_cvtpd_ps (__U,__A);
}

__m512 test_mm512_cvtph_ps (__m256i __A)
{
  // CHECK-LABEL: test_mm512_cvtph_ps 
  // CHECK: bitcast <4 x i64> %{{.*}} to <16 x i16>
  // CHECK: bitcast <16 x i16> %{{.*}} to <16 x half>
  // CHECK: fpext <16 x half> %{{.*}} to <16 x float>
  return _mm512_cvtph_ps (__A);
}

__m512 test_mm512_mask_cvtph_ps (__m512 __W, __mmask16 __U, __m256i __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtph_ps 
  // CHECK: bitcast <4 x i64> %{{.*}} to <16 x i16>
  // CHECK: bitcast <16 x i16> %{{.*}} to <16 x half>
  // CHECK: fpext <16 x half> %{{.*}} to <16 x float>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_cvtph_ps (__W,__U,__A);
}

__m512 test_mm512_maskz_cvtph_ps (__mmask16 __U, __m256i __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtph_ps 
  // CHECK: bitcast <4 x i64> %{{.*}} to <16 x i16>
  // CHECK: bitcast <16 x i16> %{{.*}} to <16 x half>
  // CHECK: fpext <16 x half> %{{.*}} to <16 x float>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_cvtph_ps (__U,__A);
}

__m256i test_mm512_mask_cvttpd_epi32 (__m256i __W, __mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_cvttpd_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.512
  return _mm512_mask_cvttpd_epi32 (__W,__U,__A);
}

__m256i test_mm512_maskz_cvttpd_epi32 (__mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvttpd_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.512
  return _mm512_maskz_cvttpd_epi32 (__U,__A);
}

__m512i test_mm512_mask_cvttps_epi32 (__m512i __W, __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_cvttps_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvttps2dq.512
  return _mm512_mask_cvttps_epi32 (__W,__U,__A);
}

__m512i test_mm512_maskz_cvttps_epi32 (__mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvttps_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvttps2dq.512
  return _mm512_maskz_cvttps_epi32 (__U,__A);
}

__m512i test_mm512_cvtps_epi32 (__m512 __A)
{
  // CHECK-LABEL: test_mm512_cvtps_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.512
  return _mm512_cvtps_epi32 (__A);
}

__m512i test_mm512_mask_cvtps_epi32 (__m512i __W, __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtps_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.512
  return _mm512_mask_cvtps_epi32 (__W,__U,__A);
}

__m512i test_mm512_maskz_cvtps_epi32 (__mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtps_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.512
  return _mm512_maskz_cvtps_epi32 (__U,__A);
}

__m256i test_mm512_cvtpd_epi32 (__m512d __A)
{
  // CHECK-LABEL: test_mm512_cvtpd_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.512
  return _mm512_cvtpd_epi32 (__A);
}

__m256i test_mm512_mask_cvtpd_epi32 (__m256i __W, __mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtpd_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.512
  return _mm512_mask_cvtpd_epi32 (__W,__U,__A);
}

__m256i test_mm512_maskz_cvtpd_epi32 (__mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtpd_epi32 
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.512
  return _mm512_maskz_cvtpd_epi32 (__U,__A);
}

__m256i test_mm512_cvtpd_epu32 (__m512d __A)
{
  // CHECK-LABEL: test_mm512_cvtpd_epu32 
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.512
  return _mm512_cvtpd_epu32 (__A);
}

__m256i test_mm512_mask_cvtpd_epu32 (__m256i __W, __mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtpd_epu32 
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.512
  return _mm512_mask_cvtpd_epu32 (__W,__U,__A);
}

__m256i test_mm512_maskz_cvtpd_epu32 (__mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtpd_epu32 
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.512
  return _mm512_maskz_cvtpd_epu32 (__U,__A);
}

__m256i test_mm512_mask_cvtps_ph(__m256i src, __mmask16 k, __m512 a) 
{
  // CHECK-LABEL: test_mm512_mask_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.512
  return _mm512_mask_cvtps_ph(src, k, a,_MM_FROUND_TO_ZERO);
}

__m256i test_mm512_maskz_cvtps_ph (__mmask16 k, __m512 a) 
{
  // CHECK-LABEL: test_mm512_maskz_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.512
  return _mm512_maskz_cvtps_ph( k, a,_MM_FROUND_TO_ZERO);
}

__m512i test_mm512_cvtps_epu32 ( __m512 __A) 
{
  // CHECK-LABEL: test_mm512_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.512
  return _mm512_cvtps_epu32(__A);
}

__m512i test_mm512_mask_cvtps_epu32 (__m512i __W, __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.512
  return _mm512_mask_cvtps_epu32( __W, __U, __A);
}
__m512i test_mm512_maskz_cvtps_epu32 (__mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_maskz_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.512
  return _mm512_maskz_cvtps_epu32( __U, __A);
}

double test_mm512_cvtsd_f64(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtsd_f64
  // CHECK: extractelement <8 x double> %{{.*}}, i32 0
  return _mm512_cvtsd_f64(A);
}

float test_mm512_cvtss_f32(__m512 A) {
  // CHECK-LABEL: test_mm512_cvtss_f32
  // CHECK: extractelement <16 x float> %{{.*}}, i32 0
  return _mm512_cvtss_f32(A);
}

__m512d test_mm512_mask_max_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B)
{
  // CHECK-LABEL: test_mm512_mask_max_pd 
  // CHECK: @llvm.x86.avx512.max.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_max_pd (__W,__U,__A,__B);
}

__m512d test_mm512_maskz_max_pd (__mmask8 __U, __m512d __A, __m512d __B)
{
  // CHECK-LABEL: test_mm512_maskz_max_pd 
  // CHECK: @llvm.x86.avx512.max.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_max_pd (__U,__A,__B);
}

__m512 test_mm512_mask_max_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B)
{
  // CHECK-LABEL: test_mm512_mask_max_ps 
  // CHECK: @llvm.x86.avx512.max.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_max_ps (__W,__U,__A,__B);
}

__m512d test_mm512_mask_max_round_pd(__m512d __W,__mmask8 __U,__m512d __A,__m512d __B)
{
  // CHECK-LABEL: test_mm512_mask_max_round_pd
  // CHECK: @llvm.x86.avx512.max.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_max_round_pd(__W,__U,__A,__B,_MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_max_round_pd(__mmask8 __U,__m512d __A,__m512d __B)
{
  // CHECK-LABEL: test_mm512_maskz_max_round_pd
  // CHECK: @llvm.x86.avx512.max.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_max_round_pd(__U,__A,__B,_MM_FROUND_NO_EXC);
}

__m512d test_mm512_max_round_pd(__m512d __A,__m512d __B)
{
  // CHECK-LABEL: test_mm512_max_round_pd
  // CHECK: @llvm.x86.avx512.max.pd.512
  return _mm512_max_round_pd(__A,__B,_MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_max_ps (__mmask16 __U, __m512 __A, __m512 __B)
{
  // CHECK-LABEL: test_mm512_maskz_max_ps 
  // CHECK: @llvm.x86.avx512.max.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_max_ps (__U,__A,__B);
}

__m512 test_mm512_mask_max_round_ps(__m512 __W,__mmask16 __U,__m512 __A,__m512 __B)
{
  // CHECK-LABEL: test_mm512_mask_max_round_ps
  // CHECK: @llvm.x86.avx512.max.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_max_round_ps(__W,__U,__A,__B,_MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_max_round_ps(__mmask16 __U,__m512 __A,__m512 __B)
{
  // CHECK-LABEL: test_mm512_maskz_max_round_ps
  // CHECK: @llvm.x86.avx512.max.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_max_round_ps(__U,__A,__B,_MM_FROUND_NO_EXC);
}

__m512 test_mm512_max_round_ps(__m512 __A,__m512 __B)
{
  // CHECK-LABEL: test_mm512_max_round_ps
  // CHECK: @llvm.x86.avx512.max.ps.512
  return _mm512_max_round_ps(__A,__B,_MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_min_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B)
{
  // CHECK-LABEL: test_mm512_mask_min_pd 
  // CHECK: @llvm.x86.avx512.min.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_min_pd (__W,__U,__A,__B);
}

__m512d test_mm512_maskz_min_pd (__mmask8 __U, __m512d __A, __m512d __B)
{
  // CHECK-LABEL: test_mm512_maskz_min_pd 
  // CHECK: @llvm.x86.avx512.min.pd.512
  return _mm512_maskz_min_pd (__U,__A,__B);
}

__m512d test_mm512_mask_min_round_pd(__m512d __W,__mmask8 __U,__m512d __A,__m512d __B)
{
  // CHECK-LABEL: test_mm512_mask_min_round_pd
  // CHECK: @llvm.x86.avx512.min.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_min_round_pd(__W,__U,__A,__B,_MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_min_round_pd(__mmask8 __U,__m512d __A,__m512d __B)
{
  // CHECK-LABEL: test_mm512_maskz_min_round_pd
  // CHECK: @llvm.x86.avx512.min.pd.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_min_round_pd(__U,__A,__B,_MM_FROUND_NO_EXC);
}

__m512d test_mm512_min_round_pd( __m512d __A,__m512d __B)
{
  // CHECK-LABEL: test_mm512_min_round_pd
  // CHECK: @llvm.x86.avx512.min.pd.512
  return _mm512_min_round_pd(__A,__B,_MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_min_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B)
{
  // CHECK-LABEL: test_mm512_mask_min_ps 
  // CHECK: @llvm.x86.avx512.min.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_min_ps (__W,__U,__A,__B);
}

__m512 test_mm512_maskz_min_ps (__mmask16 __U, __m512 __A, __m512 __B)
{
  // CHECK-LABEL: test_mm512_maskz_min_ps 
  // CHECK: @llvm.x86.avx512.min.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_min_ps (__U,__A,__B);
}

__m512 test_mm512_mask_min_round_ps(__m512 __W,__mmask16 __U,__m512 __A,__m512 __B)
{
  // CHECK-LABEL: test_mm512_mask_min_round_ps
  // CHECK: @llvm.x86.avx512.min.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_min_round_ps(__W,__U,__A,__B,_MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_min_round_ps(__mmask16 __U,__m512 __A,__m512 __B)
{
  // CHECK-LABEL: test_mm512_maskz_min_round_ps
  // CHECK: @llvm.x86.avx512.min.ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_min_round_ps(__U,__A,__B,_MM_FROUND_NO_EXC);
}

__m512 test_mm512_min_round_ps(__m512 __A,__m512 __B)
{
  // CHECK-LABEL: test_mm512_min_round_ps
  // CHECK: @llvm.x86.avx512.min.ps.512
  return _mm512_min_round_ps(__A,__B,_MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_floor_ps (__m512 __W, __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_floor_ps 
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.512
  return _mm512_mask_floor_ps (__W,__U,__A);
}

__m512d test_mm512_mask_floor_pd (__m512d __W, __mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_floor_pd 
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.512
  return _mm512_mask_floor_pd (__W,__U,__A);
}

__m512 test_mm512_mask_ceil_ps (__m512 __W, __mmask16 __U, __m512 __A)
{
  // CHECK-LABEL: test_mm512_mask_ceil_ps 
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.512
  return _mm512_mask_ceil_ps (__W,__U,__A);
}

__m512d test_mm512_mask_ceil_pd (__m512d __W, __mmask8 __U, __m512d __A)
{
  // CHECK-LABEL: test_mm512_mask_ceil_pd 
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.512
  return _mm512_mask_ceil_pd (__W,__U,__A);
}

__m512 test_mm512_mask_roundscale_ps(__m512 __W, __mmask16 __U, __m512 __A) 
{
  // CHECK-LABEL: test_mm512_mask_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.512
  return _mm512_mask_roundscale_ps(__W,__U,__A, 1);
}

__m512 test_mm512_maskz_roundscale_ps(__mmask16 __U, __m512 __A) 
{
  // CHECK-LABEL: test_mm512_maskz_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.512
  return _mm512_maskz_roundscale_ps(__U,__A, 1);
}

__m512 test_mm512_mask_roundscale_round_ps(__m512 __A,__mmask16 __U,__m512 __C)
{
  // CHECK-LABEL: test_mm512_mask_roundscale_round_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.512
  return _mm512_mask_roundscale_round_ps(__A,__U,__C,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_roundscale_round_ps(__m512 __A,__mmask16 __U) 
{
  // CHECK-LABEL: test_mm512_maskz_roundscale_round_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.512
  return _mm512_maskz_roundscale_round_ps(__U,__A,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m512 test_mm512_roundscale_round_ps(__m512 __A)
{
  // CHECK-LABEL: test_mm512_roundscale_round_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.512
  return _mm512_roundscale_round_ps(__A,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_roundscale_pd(__m512d __W, __mmask8 __U, __m512d __A) 
{
  // CHECK-LABEL: test_mm512_mask_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.512
  return _mm512_mask_roundscale_pd(__W,__U,__A, 1);
}

__m512d test_mm512_maskz_roundscale_pd(__mmask8 __U, __m512d __A) 
{
  // CHECK-LABEL: test_mm512_maskz_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.512
  return _mm512_maskz_roundscale_pd(__U,__A, 1);
}

__m512d test_mm512_mask_roundscale_round_pd(__m512d __A,__mmask8 __U,__m512d __C)
{
  // CHECK-LABEL: test_mm512_mask_roundscale_round_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.512
  return _mm512_mask_roundscale_round_pd(__A,__U,__C,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_roundscale_round_pd(__m512d __A,__mmask8 __U)
{
  // CHECK-LABEL: test_mm512_maskz_roundscale_round_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.512
  return _mm512_maskz_roundscale_round_pd(__U,__A,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m512d test_mm512_roundscale_round_pd(__m512d __A)
{
  // CHECK-LABEL: test_mm512_roundscale_round_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.512
  return _mm512_roundscale_round_pd(__A,_MM_FROUND_TO_ZERO,_MM_FROUND_NO_EXC);
}

__m512i test_mm512_max_epi32 (__m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_max_epi32 
  // CHECK:       call <16 x i32> @llvm.smax.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_max_epi32 (__A,__B);
}

__m512i test_mm512_mask_max_epi32 (__m512i __W, __mmask16 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_max_epi32 
  // CHECK:       call <16 x i32> @llvm.smax.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK:       select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32> {{.*}}
  return _mm512_mask_max_epi32 (__W,__M,__A,__B);
}

__m512i test_mm512_maskz_max_epi32 (__mmask16 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_maskz_max_epi32 
  // CHECK:       call <16 x i32> @llvm.smax.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK:       select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32> {{.*}}
  return _mm512_maskz_max_epi32 (__M,__A,__B);
}

__m512i test_mm512_max_epi64 (__m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_max_epi64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.smax.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_max_epi64 (__A,__B);
}

__m512i test_mm512_mask_max_epi64 (__m512i __W, __mmask8 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_max_epi64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.smax.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK:       select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64> {{.*}}
  return _mm512_mask_max_epi64 (__W,__M,__A,__B);
}

__m512i test_mm512_maskz_max_epi64 (__mmask8 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_maskz_max_epi64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.smax.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK:       select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64> {{.*}}
  return _mm512_maskz_max_epi64 (__M,__A,__B);
}

__m512i test_mm512_max_epu64 (__m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_max_epu64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.umax.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_max_epu64 (__A,__B);
}

__m512i test_mm512_mask_max_epu64 (__m512i __W, __mmask8 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_max_epu64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.umax.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK:       select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64> {{.*}}
  return _mm512_mask_max_epu64 (__W,__M,__A,__B);
}

__m512i test_mm512_maskz_max_epu64 (__mmask8 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_maskz_max_epu64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.umax.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK:       select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64> {{.*}}
  return _mm512_maskz_max_epu64 (__M,__A,__B);
}

__m512i test_mm512_max_epu32 (__m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_max_epu32 
  // CHECK:       call <16 x i32> @llvm.umax.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_max_epu32 (__A,__B);
}

__m512i test_mm512_mask_max_epu32 (__m512i __W, __mmask16 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_max_epu32 
  // CHECK:       call <16 x i32> @llvm.umax.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK:       select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32> {{.*}}
  return _mm512_mask_max_epu32 (__W,__M,__A,__B);
}

__m512i test_mm512_maskz_max_epu32 (__mmask16 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_maskz_max_epu32 
  // CHECK:       call <16 x i32> @llvm.umax.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK:       select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32> {{.*}}
  return _mm512_maskz_max_epu32 (__M,__A,__B);
}

__m512i test_mm512_min_epi32 (__m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_min_epi32 
  // CHECK:       call <16 x i32> @llvm.smin.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_min_epi32 (__A,__B);
}

__m512i test_mm512_mask_min_epi32 (__m512i __W, __mmask16 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_min_epi32 
  // CHECK:       call <16 x i32> @llvm.smin.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK:       select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32> {{.*}}
  return _mm512_mask_min_epi32 (__W,__M,__A,__B);
}

__m512i test_mm512_maskz_min_epi32 (__mmask16 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_maskz_min_epi32 
  // CHECK:       call <16 x i32> @llvm.smin.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK:       select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32> {{.*}}
  return _mm512_maskz_min_epi32 (__M,__A,__B);
}

__m512i test_mm512_min_epu32 (__m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_min_epu32 
  // CHECK:       call <16 x i32> @llvm.umin.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_min_epu32 (__A,__B);
}

__m512i test_mm512_mask_min_epu32 (__m512i __W, __mmask16 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_min_epu32 
  // CHECK:       call <16 x i32> @llvm.umin.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK:       select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32> {{.*}}
  return _mm512_mask_min_epu32 (__W,__M,__A,__B);
}

__m512i test_mm512_maskz_min_epu32 (__mmask16 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_maskz_min_epu32 
  // CHECK:       call <16 x i32> @llvm.umin.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK:       select <16 x i1> {{.*}}, <16 x i32> {{.*}}, <16 x i32> {{.*}}
  return _mm512_maskz_min_epu32 (__M,__A,__B);
}

__m512i test_mm512_min_epi64 (__m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_min_epi64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.smin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_min_epi64 (__A,__B);
}

__m512i test_mm512_mask_min_epi64 (__m512i __W, __mmask8 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_min_epi64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.smin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK:       select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64> {{.*}}
  return _mm512_mask_min_epi64 (__W,__M,__A,__B);
}

__m512i test_mm512_maskz_min_epi64 (__mmask8 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_maskz_min_epi64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.smin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK:       select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64> {{.*}}
  return _mm512_maskz_min_epi64 (__M,__A,__B);
}

__m512i test_mm512_min_epu64 (__m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_min_epu64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.umin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_min_epu64 (__A,__B);
}

__m512i test_mm512_mask_min_epu64 (__m512i __W, __mmask8 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_mask_min_epu64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.umin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK:       select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64> {{.*}}
  return _mm512_mask_min_epu64 (__W,__M,__A,__B);
}

__m512i test_mm512_maskz_min_epu64 (__mmask8 __M, __m512i __A, __m512i __B)
{
  // CHECK-LABEL: test_mm512_maskz_min_epu64 
  // CHECK:       call {{.*}}<8 x i64> @llvm.umin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK:       select <8 x i1> {{.*}}, <8 x i64> {{.*}}, <8 x i64> {{.*}}
  return _mm512_maskz_min_epu64 (__M,__A,__B);
}

__m512i test_mm512_mask_set1_epi32 (__m512i __O, __mmask16 __M, int __A)
{
  // CHECK-LABEL: test_mm512_mask_set1_epi32
  // CHECK: insertelement <16 x i32> poison, i32 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 15
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_set1_epi32 ( __O, __M, __A);
}

__m512i test_mm512_maskz_set1_epi32(__mmask16 __M, int __A)
{     
  // CHECK-LABEL: test_mm512_maskz_set1_epi32
  // CHECK: insertelement <16 x i32> poison, i32 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}, i32 15
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
    return _mm512_maskz_set1_epi32(__M, __A);
}


__m512i test_mm512_set_epi8(char e63, char e62, char e61, char e60, char e59,
    char e58, char e57, char e56, char e55, char e54, char e53, char e52,
    char e51, char e50, char e49, char e48, char e47, char e46, char e45,
    char e44, char e43, char e42, char e41, char e40, char e39, char e38,
    char e37, char e36, char e35, char e34, char e33, char e32, char e31,
    char e30, char e29, char e28, char e27, char e26, char e25, char e24,
    char e23, char e22, char e21, char e20, char e19, char e18, char e17,
    char e16, char e15, char e14, char e13, char e12, char e11, char e10,
    char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2,
    char e1, char e0) {

  //CHECK-LABEL: test_mm512_set_epi8
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  //CHECK: load i8, ptr %{{.*}}, align 1
  return _mm512_set_epi8(e63, e62, e61, e60, e59, e58, e57, e56, e55, e54,
      e53, e52, e51, e50, e49, e48,e47, e46, e45, e44, e43, e42, e41, e40,
      e39, e38, e37, e36, e35, e34, e33, e32,e31, e30, e29, e28, e27, e26,
      e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12,
      e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0);
}

__m512i test_mm512_set_epi16(short e31, short e30, short e29, short e28,
    short e27, short e26, short e25, short e24, short e23, short e22,
    short e21, short e20, short e19, short e18, short e17,
    short e16, short e15, short e14, short e13, short e12,
    short e11, short e10, short e9, short e8, short e7,
    short e6, short e5, short e4, short e3, short e2, short e1, short e0) {
  //CHECK-LABEL: test_mm512_set_epi16
  //CHECK: insertelement{{.*}}i32 0
  //CHECK: insertelement{{.*}}i32 1
  //CHECK: insertelement{{.*}}i32 2
  //CHECK: insertelement{{.*}}i32 3
  //CHECK: insertelement{{.*}}i32 4
  //CHECK: insertelement{{.*}}i32 5
  //CHECK: insertelement{{.*}}i32 6
  //CHECK: insertelement{{.*}}i32 7
  //CHECK: insertelement{{.*}}i32 8
  //CHECK: insertelement{{.*}}i32 9
  //CHECK: insertelement{{.*}}i32 10
  //CHECK: insertelement{{.*}}i32 11
  //CHECK: insertelement{{.*}}i32 12
  //CHECK: insertelement{{.*}}i32 13
  //CHECK: insertelement{{.*}}i32 14
  //CHECK: insertelement{{.*}}i32 15
  //CHECK: insertelement{{.*}}i32 16
  //CHECK: insertelement{{.*}}i32 17
  //CHECK: insertelement{{.*}}i32 18
  //CHECK: insertelement{{.*}}i32 19
  //CHECK: insertelement{{.*}}i32 20
  //CHECK: insertelement{{.*}}i32 21
  //CHECK: insertelement{{.*}}i32 22
  //CHECK: insertelement{{.*}}i32 23
  //CHECK: insertelement{{.*}}i32 24
  //CHECK: insertelement{{.*}}i32 25
  //CHECK: insertelement{{.*}}i32 26
  //CHECK: insertelement{{.*}}i32 27
  //CHECK: insertelement{{.*}}i32 28
  //CHECK: insertelement{{.*}}i32 29
  //CHECK: insertelement{{.*}}i32 30
  //CHECK: insertelement{{.*}}i32 31
  return _mm512_set_epi16(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22,
      e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7,
      e6, e5, e4, e3, e2, e1, e0);

}
__m512i test_mm512_set_epi32 (int __A, int __B, int __C, int __D,
               int __E, int __F, int __G, int __H,
               int __I, int __J, int __K, int __L,
               int __M, int __N, int __O, int __P)
{
 //CHECK-LABEL: test_mm512_set_epi32
 //CHECK: insertelement{{.*}}i32 0
 //CHECK: insertelement{{.*}}i32 1
 //CHECK: insertelement{{.*}}i32 2
 //CHECK: insertelement{{.*}}i32 3
 //CHECK: insertelement{{.*}}i32 4
 //CHECK: insertelement{{.*}}i32 5
 //CHECK: insertelement{{.*}}i32 6
 //CHECK: insertelement{{.*}}i32 7
 //CHECK: insertelement{{.*}}i32 8
 //CHECK: insertelement{{.*}}i32 9
 //CHECK: insertelement{{.*}}i32 10
 //CHECK: insertelement{{.*}}i32 11
 //CHECK: insertelement{{.*}}i32 12
 //CHECK: insertelement{{.*}}i32 13
 //CHECK: insertelement{{.*}}i32 14
 //CHECK: insertelement{{.*}}i32 15
 return _mm512_set_epi32( __A, __B, __C, __D,__E, __F, __G, __H,
              __I, __J, __K, __L,__M, __N, __O, __P);
}

__m512i test_mm512_setr_epi32 (int __A, int __B, int __C, int __D,
               int __E, int __F, int __G, int __H,
               int __I, int __J, int __K, int __L,
               int __M, int __N, int __O, int __P)
{
 //CHECK-LABEL: test_mm512_setr_epi32
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: load{{.*}}%{{.*}}, align 4
 //CHECK: insertelement{{.*}}i32 0
 //CHECK: insertelement{{.*}}i32 1
 //CHECK: insertelement{{.*}}i32 2
 //CHECK: insertelement{{.*}}i32 3
 //CHECK: insertelement{{.*}}i32 4
 //CHECK: insertelement{{.*}}i32 5
 //CHECK: insertelement{{.*}}i32 6
 //CHECK: insertelement{{.*}}i32 7
 //CHECK: insertelement{{.*}}i32 8
 //CHECK: insertelement{{.*}}i32 9
 //CHECK: insertelement{{.*}}i32 10
 //CHECK: insertelement{{.*}}i32 11
 //CHECK: insertelement{{.*}}i32 12
 //CHECK: insertelement{{.*}}i32 13
 //CHECK: insertelement{{.*}}i32 14
 //CHECK: insertelement{{.*}}i32 15
 return _mm512_setr_epi32( __A, __B, __C, __D,__E, __F, __G, __H,
              __I, __J, __K, __L,__M, __N, __O, __P);
}

__m512i test_mm512_mask_set1_epi64 (__m512i __O, __mmask8 __M, long long __A)
{
  // CHECK-LABEL: test_mm512_mask_set1_epi64
  // CHECK: insertelement <8 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 7
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_set1_epi64 (__O, __M, __A);
}

__m512i test_mm512_maskz_set1_epi64 (__mmask8 __M, long long __A)
{
  // CHECK-LABEL: test_mm512_maskz_set1_epi64
  // CHECK: insertelement <8 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 7
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_set1_epi64 (__M, __A);
}


__m512i test_mm512_set_epi64 (long long __A, long long __B, long long __C,
                              long long __D, long long __E, long long __F,
                              long long __G, long long __H)
{
    //CHECK-LABEL: test_mm512_set_epi64
    //CHECK: insertelement{{.*}}i32 0
    //CHECK: insertelement{{.*}}i32 1
    //CHECK: insertelement{{.*}}i32 2
    //CHECK: insertelement{{.*}}i32 3
    //CHECK: insertelement{{.*}}i32 4
    //CHECK: insertelement{{.*}}i32 5
    //CHECK: insertelement{{.*}}i32 6
    //CHECK: insertelement{{.*}}i32 7
  return _mm512_set_epi64(__A, __B, __C, __D, __E, __F, __G, __H );
}

__m512i test_mm512_setr_epi64 (long long __A, long long __B, long long __C,
                              long long __D, long long __E, long long __F,
                              long long __G, long long __H)
{
    //CHECK-LABEL: test_mm512_setr_epi64
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: insertelement{{.*}}i32 0
    //CHECK: insertelement{{.*}}i32 1
    //CHECK: insertelement{{.*}}i32 2
    //CHECK: insertelement{{.*}}i32 3
    //CHECK: insertelement{{.*}}i32 4
    //CHECK: insertelement{{.*}}i32 5
    //CHECK: insertelement{{.*}}i32 6
    //CHECK: insertelement{{.*}}i32 7
  return _mm512_setr_epi64(__A, __B, __C, __D, __E, __F, __G, __H );
}

__m512d test_mm512_set_pd (double __A, double __B, double __C, double __D,
                           double __E, double __F, double __G, double __H)
{
    //CHECK-LABEL: test_mm512_set_pd
    //CHECK: insertelement{{.*}}i32 0
    //CHECK: insertelement{{.*}}i32 1
    //CHECK: insertelement{{.*}}i32 2
    //CHECK: insertelement{{.*}}i32 3
    //CHECK: insertelement{{.*}}i32 4
    //CHECK: insertelement{{.*}}i32 5
    //CHECK: insertelement{{.*}}i32 6
    //CHECK: insertelement{{.*}}i32 7
  return _mm512_set_pd( __A, __B, __C, __D, __E, __F, __G, __H);
}

__m512d test_mm512_setr_pd (double __A, double __B, double __C, double __D,
                           double __E, double __F, double __G, double __H)
{
    //CHECK-LABEL: test_mm512_setr_pd
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: load{{.*}}%{{.*}}, align 8
    //CHECK: insertelement{{.*}}i32 0
    //CHECK: insertelement{{.*}}i32 1
    //CHECK: insertelement{{.*}}i32 2
    //CHECK: insertelement{{.*}}i32 3
    //CHECK: insertelement{{.*}}i32 4
    //CHECK: insertelement{{.*}}i32 5
    //CHECK: insertelement{{.*}}i32 6
    //CHECK: insertelement{{.*}}i32 7
  return _mm512_setr_pd( __A, __B, __C, __D, __E, __F, __G, __H);
}

__m512 test_mm512_set_ps (float __A, float __B, float __C, float __D,
                          float __E, float __F, float __G, float __H,
                          float __I, float __J, float __K, float __L,
                          float __M, float __N, float __O, float __P)
{
    //CHECK-LABEL: test_mm512_set_ps
    //CHECK: insertelement{{.*}}i32 0
    //CHECK: insertelement{{.*}}i32 1
    //CHECK: insertelement{{.*}}i32 2
    //CHECK: insertelement{{.*}}i32 3
    //CHECK: insertelement{{.*}}i32 4
    //CHECK: insertelement{{.*}}i32 5
    //CHECK: insertelement{{.*}}i32 6
    //CHECK: insertelement{{.*}}i32 7
    //CHECK: insertelement{{.*}}i32 8
    //CHECK: insertelement{{.*}}i32 9
    //CHECK: insertelement{{.*}}i32 10
    //CHECK: insertelement{{.*}}i32 11
    //CHECK: insertelement{{.*}}i32 12
    //CHECK: insertelement{{.*}}i32 13
    //CHECK: insertelement{{.*}}i32 14
    //CHECK: insertelement{{.*}}i32 15
    return _mm512_set_ps( __A, __B, __C, __D, __E, __F, __G, __H,
                          __I, __J, __K, __L, __M, __N, __O, __P);
}

__m512i test_mm512_mask_abs_epi64 (__m512i __W, __mmask8 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_mask_abs_epi64 
  // CHECK: [[ABS:%.*]] = call {{.*}}<8 x i64> @llvm.abs.v8i64(<8 x i64> %{{.*}}, i1 false)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> [[ABS]], <8 x i64> %{{.*}}
  return _mm512_mask_abs_epi64 (__W,__U,__A);
}

__m512i test_mm512_maskz_abs_epi64 (__mmask8 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_maskz_abs_epi64 
  // CHECK: [[ABS:%.*]] = call {{.*}}<8 x i64> @llvm.abs.v8i64(<8 x i64> %{{.*}}, i1 false)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> [[ABS]], <8 x i64> %{{.*}}
  return _mm512_maskz_abs_epi64 (__U,__A);
}

__m512i test_mm512_mask_abs_epi32 (__m512i __W, __mmask16 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_mask_abs_epi32
  // CHECK: [[ABS:%.*]] = call <16 x i32> @llvm.abs.v16i32(<16 x i32> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <16 x i32> [[ABS]] to <8 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <8 x i64> [[TMP]] to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> [[ABS]], <16 x i32> %{{.*}}
  return _mm512_mask_abs_epi32 (__W,__U,__A);
}

__m512i test_mm512_maskz_abs_epi32 (__mmask16 __U, __m512i __A)
{
  // CHECK-LABEL: test_mm512_maskz_abs_epi32
  // CHECK: [[ABS:%.*]] = call <16 x i32> @llvm.abs.v16i32(<16 x i32> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <16 x i32> [[ABS]] to <8 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <8 x i64> [[TMP]] to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> [[ABS]], <16 x i32> %{{.*}}
  return _mm512_maskz_abs_epi32 (__U,__A);
}

__m512 test_mm512_setr_ps (float __A, float __B, float __C, float __D,
                          float __E, float __F, float __G, float __H,
                          float __I, float __J, float __K, float __L,
                          float __M, float __N, float __O, float __P)
{
    //CHECK-LABEL: test_mm512_setr_ps
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: load{{.*}}%{{.*}}, align 4
    //CHECK: insertelement{{.*}}i32 0
    //CHECK: insertelement{{.*}}i32 1
    //CHECK: insertelement{{.*}}i32 2
    //CHECK: insertelement{{.*}}i32 3
    //CHECK: insertelement{{.*}}i32 4
    //CHECK: insertelement{{.*}}i32 5
    //CHECK: insertelement{{.*}}i32 6
    //CHECK: insertelement{{.*}}i32 7
    //CHECK: insertelement{{.*}}i32 8
    //CHECK: insertelement{{.*}}i32 9
    //CHECK: insertelement{{.*}}i32 10
    //CHECK: insertelement{{.*}}i32 11
    //CHECK: insertelement{{.*}}i32 12
    //CHECK: insertelement{{.*}}i32 13
    //CHECK: insertelement{{.*}}i32 14
    //CHECK: insertelement{{.*}}i32 15
    return _mm512_setr_ps( __A, __B, __C, __D, __E, __F, __G, __H,
                          __I, __J, __K, __L, __M, __N, __O, __P);
}

int test_mm_cvtss_i32(__m128 A) {
  // CHECK-LABEL: test_mm_cvtss_i32
  // CHECK: call {{.*}}i32 @llvm.x86.sse.cvtss2si(<4 x float> %{{.*}})
  return _mm_cvtss_i32(A);
}

#ifdef __x86_64__
long long test_mm_cvtss_i64(__m128 A) {
  // CHECK-LABEL: test_mm_cvtss_i64
  // CHECK: call {{.*}}i64 @llvm.x86.sse.cvtss2si64(<4 x float> %{{.*}})
  return _mm_cvtss_i64(A);
}
#endif

__m128d test_mm_cvti32_sd(__m128d A, int B) {
  // CHECK-LABEL: test_mm_cvti32_sd
  // CHECK: sitofp i32 %{{.*}} to double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_cvti32_sd(A, B);
}

#ifdef __x86_64__
__m128d test_mm_cvti64_sd(__m128d A, long long B) {
  // CHECK-LABEL: test_mm_cvti64_sd
  // CHECK: sitofp i64 %{{.*}} to double
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  return _mm_cvti64_sd(A, B);
}
#endif

__m128 test_mm_cvti32_ss(__m128 A, int B) {
  // CHECK-LABEL: test_mm_cvti32_ss
  // CHECK: sitofp i32 %{{.*}} to float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_cvti32_ss(A, B);
}

#ifdef __x86_64__
__m128 test_mm_cvti64_ss(__m128 A, long long B) {
  // CHECK-LABEL: test_mm_cvti64_ss
  // CHECK: sitofp i64 %{{.*}} to float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_cvti64_ss(A, B);
}
#endif

int test_mm_cvtsd_i32(__m128d A) {
  // CHECK-LABEL: test_mm_cvtsd_i32
  // CHECK: call {{.*}}i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %{{.*}})
  return _mm_cvtsd_i32(A);
}

#ifdef __x86_64__
long long test_mm_cvtsd_i64(__m128d A) {
  // CHECK-LABEL: test_mm_cvtsd_i64
  // CHECK: call {{.*}}i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %{{.*}})
  return _mm_cvtsd_i64(A);
}
#endif

__m128d test_mm_mask_cvtss_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_cvtss_sd
  // CHECK: @llvm.x86.avx512.mask.cvtss2sd.round
  return _mm_mask_cvtss_sd(__W, __U, __A, __B); 
}

__m128d test_mm_maskz_cvtss_sd( __mmask8 __U, __m128d __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_cvtss_sd
  // CHECK: @llvm.x86.avx512.mask.cvtss2sd.round
  return _mm_maskz_cvtss_sd( __U, __A, __B); 
}

__m128 test_mm_mask_cvtsd_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_cvtsd_ss
  // CHECK: @llvm.x86.avx512.mask.cvtsd2ss.round
  return _mm_mask_cvtsd_ss(__W, __U, __A, __B); 
}

__m128 test_mm_maskz_cvtsd_ss(__mmask8 __U, __m128 __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_cvtsd_ss
  // CHECK: @llvm.x86.avx512.mask.cvtsd2ss.round
  return _mm_maskz_cvtsd_ss(__U, __A, __B); 
}

__m512i test_mm512_setzero_epi32(void)
{
  // CHECK-LABEL: test_mm512_setzero_epi32
  // CHECK: zeroinitializer
  return _mm512_setzero_epi32();
}
TEST_CONSTEXPR(match_m512i(_mm512_setzero_epi32(), 0, 0, 0, 0, 0, 0, 0, 0));

__m512 test_mm512_setzero(void)
{
  // CHECK-LABEL: test_mm512_setzero
  // CHECK: zeroinitializer
  return _mm512_setzero();
}
TEST_CONSTEXPR(match_m512(_mm512_setzero(), +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f));

__m512i test_mm512_setzero_si512(void)
{
  // CHECK-LABEL: test_mm512_setzero_si512
  // CHECK: zeroinitializer
  return _mm512_setzero_si512();
}
TEST_CONSTEXPR(match_m512i(_mm512_setzero_si512(), 0, 0, 0, 0, 0, 0, 0, 0));

__m512 test_mm512_setzero_ps(void)
{
  // CHECK-LABEL: test_mm512_setzero_ps
  // CHECK: zeroinitializer
  return _mm512_setzero_ps();
}
TEST_CONSTEXPR(match_m512(_mm512_setzero_ps(), +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f));

__m512d test_mm512_setzero_pd(void)
{
  // CHECK-LABEL: test_mm512_setzero_pd
  // CHECK: zeroinitializer
  return _mm512_setzero_pd();
}
TEST_CONSTEXPR(match_m512d(_mm512_setzero_pd(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

__mmask16 test_mm512_int2mask(int __a)
{
  // CHECK-LABEL: test_mm512_int2mask
  // CHECK: trunc i32 %{{.*}} to i16
  return _mm512_int2mask(__a);
}

int test_mm512_mask2int(__mmask16 __a)
{
  // CHECK-LABEL: test_mm512_mask2int
  // CHECK: zext i16 %{{.*}} to i32
  return _mm512_mask2int(__a);
}

__m128 test_mm_mask_move_ss (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B)
{
  // CHECK-LABEL: test_mm_mask_move_ss
  // CHECK: [[EXT:%.*]] = extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: insertelement <4 x float> %{{.*}}, float [[EXT]], i32 0
  // CHECK: [[A:%.*]] = extractelement <4 x float> [[VEC:%.*]], i64 0
  // CHECK-NEXT: [[B:%.*]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.*]] = select i1 %{{.*}}, float [[A]], float [[B]]
  // CHECK-NEXT: insertelement <4 x float> [[VEC]], float [[SEL]], i64 0
  return _mm_mask_move_ss ( __W,  __U,  __A,  __B);
}

__m128 test_mm_maskz_move_ss (__mmask8 __U, __m128 __A, __m128 __B)
{
  // CHECK-LABEL: test_mm_maskz_move_ss
  // CHECK: [[EXT:%.*]] = extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: insertelement <4 x float> %{{.*}}, float [[EXT]], i32 0
  // CHECK: [[A:%.*]] = extractelement <4 x float> [[VEC:%.*]], i64 0
  // CHECK-NEXT: [[B:%.*]] = extractelement <4 x float> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.*]] = select i1 %{{.*}}, float [[A]], float [[B]]
  // CHECK-NEXT: insertelement <4 x float> [[VEC]], float [[SEL]], i64 0
  return _mm_maskz_move_ss (__U, __A, __B);
}

__m128d test_mm_mask_move_sd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B)
{
  // CHECK-LABEL: test_mm_mask_move_sd
  // CHECK: [[EXT:%.*]] = extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double [[EXT]], i32 0
  // CHECK: [[A:%.*]] = extractelement <2 x double> [[VEC:%.*]], i64 0
  // CHECK-NEXT: [[B:%.*]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.*]] = select i1 %{{.*}}, double [[A]], double [[B]]
  // CHECK-NEXT: insertelement <2 x double> [[VEC]], double [[SEL]], i64 0
  return _mm_mask_move_sd ( __W,  __U,  __A,  __B);
}

__m128d test_mm_maskz_move_sd (__mmask8 __U, __m128d __A, __m128d __B)
{
  // CHECK-LABEL: test_mm_maskz_move_sd
  // CHECK: [[EXT:%.*]] = extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double [[EXT]], i32 0
  // CHECK: [[A:%.*]] = extractelement <2 x double> [[VEC:%.*]], i64 0
  // CHECK-NEXT: [[B:%.*]] = extractelement <2 x double> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.*]] = select i1 %13, double [[A]], double [[B]]
  // CHECK-NEXT: insertelement <2 x double> [[VEC]], double [[SEL]], i64 0
  return _mm_maskz_move_sd (__U, __A, __B);
}

void test_mm_mask_store_ss(float * __P, __mmask8 __U, __m128 __A)
{
  // CHECK-LABEL: test_mm_mask_store_ss
  // CHECK: call void @llvm.masked.store.v4f32.p0(<4 x float> %{{.*}}, ptr %{{.*}}, i32 1, <4 x i1> %{{.*}})
  _mm_mask_store_ss(__P, __U, __A);
}

void test_mm_mask_store_sd(double * __P, __mmask8 __U, __m128d __A)
{
  // CHECK-LABEL: test_mm_mask_store_sd
  // CHECK: call void @llvm.masked.store.v2f64.p0(<2 x double> %{{.*}}, ptr %{{.*}}, i32 1, <2 x i1> %{{.*}})
  _mm_mask_store_sd(__P, __U, __A);
}

__m128 test_mm_mask_load_ss(__m128 __A, __mmask8 __U, const float* __W)
{
  // CHECK-LABEL: test_mm_mask_load_ss
  // CHECK: call {{.*}}<4 x float> @llvm.masked.load.v4f32.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_load_ss(__A, __U, __W);
}

__m128 test_mm_maskz_load_ss (__mmask8 __U, const float * __W)
{
  // CHECK-LABEL: test_mm_maskz_load_ss
  // CHECK: call {{.*}}<4 x float> @llvm.masked.load.v4f32.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_load_ss (__U, __W);
}

__m128d test_mm_mask_load_sd (__m128d __A, __mmask8 __U, const double * __W)
{
  // CHECK-LABEL: test_mm_mask_load_sd
  // CHECK: call {{.*}}<2 x double> @llvm.masked.load.v2f64.p0(ptr %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_load_sd (__A, __U, __W);
}

__m128d test_mm_maskz_load_sd (__mmask8 __U, const double * __W)
{
  // CHECK-LABEL: test_mm_maskz_load_sd
  // CHECK: call {{.*}}<2 x double> @llvm.masked.load.v2f64.p0(ptr %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_load_sd (__U, __W);
}

__m512d test_mm512_abs_pd(__m512d a){
  // CHECK-LABEL: test_mm512_abs_pd
  // CHECK: and <8 x i64> 
  return _mm512_abs_pd(a);
}

__m512d test_mm512_mask_abs_pd (__m512d __W, __mmask8 __U, __m512d __A){
  // CHECK-LABEL: test_mm512_mask_abs_pd 
  // CHECK: %[[AND_RES:.*]] = and <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x i64> %[[AND_RES]], <8 x i64> %{{.*}}
  return _mm512_mask_abs_pd (__W,__U,__A);
}

__m512 test_mm512_abs_ps(__m512 a){
  // CHECK-LABEL: test_mm512_abs_ps
  // CHECK: and <16 x i32> 
  return _mm512_abs_ps(a);
}

__m512 test_mm512_mask_abs_ps(__m512 __W, __mmask16 __U, __m512 __A){
  // CHECK-LABEL: test_mm512_mask_abs_ps
  // CHECK: and <16 x i32> 
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_abs_ps( __W, __U, __A);
}

__m512d test_mm512_zextpd128_pd512(__m128d A) {
  // CHECK-LABEL: test_mm512_zextpd128_pd512
  // CHECK: store <2 x double> zeroinitializer
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  return _mm512_zextpd128_pd512(A);
}

__m512d test_mm512_zextpd256_pd512(__m256d A) {
  // CHECK-LABEL: test_mm512_zextpd256_pd512
  // CHECK: store <4 x double> zeroinitializer
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_zextpd256_pd512(A);
}

__m512 test_mm512_zextps128_ps512(__m128 A) {
  // CHECK-LABEL: test_mm512_zextps128_ps512
  // CHECK: store <4 x float> zeroinitializer
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7>
  return _mm512_zextps128_ps512(A);
}

__m512 test_mm512_zextps256_ps512(__m256 A) {
  // CHECK-LABEL: test_mm512_zextps256_ps512
  // CHECK: store <8 x float> zeroinitializer
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_zextps256_ps512(A);
}

__m512i test_mm512_zextsi128_si512(__m128i A) {
  // CHECK-LABEL: test_mm512_zextsi128_si512
  // CHECK: store <2 x i64> zeroinitializer
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  return _mm512_zextsi128_si512(A);
}

__m512i test_mm512_zextsi256_si512(__m256i A) {
  // CHECK-LABEL: test_mm512_zextsi256_si512
  // CHECK: store <4 x i64> zeroinitializer
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_zextsi256_si512(A);
}

__m512d test_mm512_i32logather_pd(__m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i32logather_pd
  // CHECK: @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_i32logather_pd(__index, __addr, 2);
}

__m512d test_mm512_mask_i32logather_pd(__m512d __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i32logather_pd
  // CHECK: @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_mask_i32logather_pd(__v1_old, __mask, __index, __addr, 2);
}

void test_mm512_i32loscatter_pd(void *__addr, __m512i __index, __m512d __v1) {
  // CHECK-LABEL: test_mm512_i32loscatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatter.dpd.512
  return _mm512_i32loscatter_pd(__addr, __index, __v1, 2);
}

void test_mm512_mask_i32loscatter_pd(void *__addr, __mmask8 __mask, __m512i __index, __m512d __v1) {
  // CHECK-LABEL: test_mm512_mask_i32loscatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatter.dpd.512
  return _mm512_mask_i32loscatter_pd(__addr, __mask, __index, __v1, 2);
}

__m512i test_mm512_i32logather_epi64(__m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_i32logather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_i32logather_epi64(__index, __addr, 2);
}

__m512i test_mm512_mask_i32logather_epi64(__m512i __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CHECK-LABEL: test_mm512_mask_i32logather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_mask_i32logather_epi64(__v1_old, __mask, __index, __addr, 2);
}

void test_mm512_i32loscatter_epi64(void *__addr, __m512i __index, __m512i __v1) {
  // CHECK-LABEL: test_mm512_i32loscatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatter.dpq.512
  _mm512_i32loscatter_epi64(__addr, __index, __v1, 2);
}

void test_mm512_mask_i32loscatter_epi64(void *__addr, __mmask8 __mask, __m512i __index, __m512i __v1) {
  // CHECK-LABEL: test_mm512_mask_i32loscatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatter.dpq.512
  _mm512_mask_i32loscatter_epi64(__addr, __mask, __index, __v1, 2);
}
