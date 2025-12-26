// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s

#include <immintrin.h>

__m128 test_vcvtph2ps_mask(__m128i a, __m128 src, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps_mask
  // CIR: cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<8 x !s16i>
  // CIR: cir.load {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR: cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
  // CIR: cir.const #cir.poison : !cir.vector<8 x !s16i>
  // CIR: cir.vec.shuffle({{.*}}) {{.*}} : !cir.vector<4 x !s16i>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<4 x !s16i> -> !cir.vector<4 x !cir.f16>
  // CIR: cir.cast floating {{.*}} : !cir.vector<4 x !cir.f16> -> !cir.vector<4 x !cir.float>
  // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.{{(bool|int<s, 1>)}}>
  // CIR: cir.vec.shuffle({{.*}}) {{.*}} : !cir.vector<4 x !cir.{{(bool|int<s, 1>)}}>
  // CIR: cir.{{(select if|vec.ternary)}}{{.*}}

  // LLVM-LABEL: @test_vcvtph2ps_mask
  // LLVM: bitcast <2 x i64> {{.*}} to <8 x i16>
  // LLVM: shufflevector <8 x i16> {{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: bitcast <4 x i16> {{.*}} to <4 x half>
  // LLVM: fpext <4 x half> {{.*}} to <4 x float>
  // LLVM: shufflevector <8 x i1> {{.*}}, <8 x i1> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: icmp ne <4 x i1> {{.*}}, zeroinitializer
  // LLVM: select <4 x i1> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}
  // LLVM: ret <4 x float> {{.*}}
  
  // OGCG-LABEL: @test_vcvtph2ps_mask
  // OGCG: bitcast <2 x i64> {{.*}} to <8 x i16>
  // OGCG: shufflevector <8 x i16> {{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: fpext <4 x half> {{.*}} to <4 x float>
  // OGCG: shufflevector <8 x i1> {{.*}}, <8 x i1> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: icmp ne <4 x i1> {{.*}}, zeroinitializer
  // OGCG: select <4 x i1> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps_mask((__v8hi)a, src, k);
}

__m256 test_vcvtph2ps256_mask(__m128i a, __m256 src, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps256_mask
  // CIR: cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<8 x !s16i>
  // CIR: cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !cir.float>>, !cir.vector<8 x !cir.float>
  // CIR: cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !s16i> -> !cir.vector<8 x !cir.f16>
  // CIR: cir.cast floating {{.*}} : !cir.vector<8 x !cir.f16> -> !cir.vector<8 x !cir.float>
  // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.{{(bool|int<s, 1>)}}>
  // CIR: cir.{{(select if|vec.ternary)}}{{.*}}cir.vector<8 x !cir.{{(bool|int<s, 1>)}}>, !cir.vector<8 x !cir.float>

  // LLVM-LABEL: @test_vcvtph2ps256_mask
  // LLVM: bitcast <2 x i64> {{.*}} to <8 x i16>
  // LLVM: bitcast <8 x i16> {{.*}} to <8 x half>
  // LLVM: fpext <8 x half> {{.*}} to <8 x float>
  // LLVM: bitcast i8 {{.*}} to <8 x i1>
  // LLVM: icmp ne <8 x i1> {{.*}}, zeroinitializer
  // LLVM: select <8 x i1> {{.*}}, <8 x float> {{.*}}, <8 x float> {{.*}}

  // OGCG-LABEL: @test_vcvtph2ps256_mask
  // OGCG: bitcast <2 x i64> {{.*}} to <8 x i16>
  // OGCG: bitcast <8 x i16> {{.*}} to <8 x half>
  // OGCG: fpext <8 x half> {{.*}} to <8 x float>
  // OGCG: bitcast i8 {{.*}} to <8 x i1>
  // OGCG: icmp ne <8 x i1> {{.*}}, zeroinitializer
  // OGCG: select <8 x i1> {{.*}}, <8 x float> {{.*}}, <8 x float> {{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps256_mask((__v8hi)a, src, k);
}

__m512 test_vcvtph2ps512_mask(__m256i a, __m512 src, __mmask16 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps512_mask
  // CIR: cir.load {{.*}} : !cir.ptr<!cir.vector<4 x !s64i>>, !cir.vector<4 x !s64i>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<4 x !s64i> -> !cir.vector<16 x !s16i>
  // CIR: cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !cir.float>>, !cir.vector<16 x !cir.float>
  // CIR: cir.load {{.*}} : !cir.ptr<!u16i>, !u16i
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !s16i> -> !cir.vector<16 x !cir.f16>
  // CIR: cir.cast floating {{.*}} : !cir.vector<16 x !cir.f16> -> !cir.vector<16 x !cir.float>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.{{(bool|int<s, 1>)}}>
  // CIR: cir.{{(select if|vec.ternary)}}{{.*}}cir.vector<16 x !cir.{{(bool|int<s, 1>)}}>, !cir.vector<16 x !cir.float>

  // LLVM-LABEL: @test_vcvtph2ps512_mask
  // LLVM: bitcast <4 x i64> {{.*}} to <16 x i16>
  // LLVM: bitcast <16 x i16> {{.*}} to <16 x half>
  // LLVM: fpext <16 x half> {{.*}} to <16 x float>
  // LLVM: bitcast i16 {{.*}} to <16 x i1>
  // LLVM: icmp ne <16 x i1> {{.*}}, zeroinitializer
  // LLVM: select <16 x i1> {{.*}}, <16 x float> {{.*}}, <16 x float> {{.*}}

  // OGCG-LABEL: @test_vcvtph2ps512_mask
  // OGCG: bitcast <4 x i64> {{.*}} to <16 x i16>
  // OGCG: bitcast <16 x i16> {{.*}} to <16 x half>
  // OGCG: fpext <16 x half> {{.*}} to <16 x float>
  // OGCG: bitcast i16 {{.*}} to <16 x i1>
  // OGCG: icmp ne <16 x i1> {{.*}}, zeroinitializer
  // OGCG: select <16 x i1> {{.*}}, <16 x float> {{.*}}, <16 x float> {{.*}}
  typedef short __v16hi __attribute__((__vector_size__(32)));
  return __builtin_ia32_vcvtph2ps512_mask((__v16hi)a, src, k, 4);
}

__m128 test_vcvtph2ps_maskz(__m128i a, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps_maskz
  // CIR: %{{.*}} = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<8 x !s16i>
  // CIR: %{{.*}} = cir.call @_mm_setzero_ps() : () -> !cir.vector<4 x !cir.float>
  // CIR: %{{.*}} = cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
  // CIR: %{{.*}} = cir.const #cir.poison : !cir.vector<8 x !s16i>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) {indices = [0, 1, 2, 3]} : !cir.vector<4 x !s16i>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<4 x !s16i> -> !cir.vector<4 x !cir.f16>
  // CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.vector<4 x !cir.f16> -> !cir.vector<4 x !cir.float>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.{{(bool|int<s, 1>)}}>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.{{(bool|int<s, 1>)}}>) {indices = [0, 1, 2, 3]} : !cir.vector<4 x !cir.{{(bool|int<s, 1>)}}>
  // CIR: cir.{{(select if|vec.ternary)}}{{.*}}%{{.*}}, %{{.*}}, %{{.*}}

  // LLVM-LABEL: @test_vcvtph2ps_maskz
  // LLVM: %{{.*}} = bitcast <2 x i64> {{.*}} to <8 x i16>
  // LLVM: %{{.*}} = shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %{{.*}} = bitcast <4 x i16> %{{.*}} to <4 x half>
  // LLVM: %{{.*}} = fpext <4 x half> %{{.*}} to <4 x float>
  // LLVM: %{{.*}} = bitcast i8 {{.*}} to <8 x i1>
  // LLVM: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> {{.*}}
  // LLVM: ret <4 x float> {{.*}}

  // OGCG-LABEL: @test_vcvtph2ps_maskz
  // OGCG: %{{.*}} = bitcast <2 x i64> {{.*}} to <8 x i16>
  // OGCG: %{{.*}} = shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: %{{.*}} = bitcast <4 x i16> %{{.*}} to <4 x half>
  // OGCG: %{{.*}} = fpext <4 x half> %{{.*}} to <4 x float>
  // OGCG: %{{.*}} = bitcast i8 {{.*}} to <8 x i1>
  // OGCG: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> {{.*}}
  // OGCG: ret <4 x float> {{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps_mask((__v8hi)a, _mm_setzero_ps(), k);
}

__m256 test_vcvtph2ps256_maskz(__m128i a, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps256_maskz
  // CIR: %{{.*}} = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<8 x !s16i>
  // CIR: %{{.*}} = cir.call @_mm256_setzero_ps() : () -> !cir.vector<8 x !cir.float>
  // CIR: %{{.*}} = cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s16i> -> !cir.vector<8 x !cir.f16>
  // CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.vector<8 x !cir.f16> -> !cir.vector<8 x !cir.float>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.{{(bool|int<s, 1>)}}>
  // CIR: cir.{{(select if|vec.ternary)}}{{.*}}%{{.*}}, %{{.*}}, %{{.*}}

  // LLVM-LABEL: @test_vcvtph2ps256_maskz
  // LLVM: %{{.*}} = bitcast <2 x i64> {{.*}} to <8 x i16>
  // LLVM: %{{.*}} = bitcast <8 x i16> %{{.*}} to <8 x half>
  // LLVM: %{{.*}} = fpext <8 x half> %{{.*}} to <8 x float>
  // LLVM: %{{.*}} = bitcast i8 {{.*}} to <8 x i1>
  // LLVM: %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> {{.*}}
  // LLVM: ret <8 x float> {{.*}}

  // OGCG-LABEL: @test_vcvtph2ps256_maskz
  // OGCG: %{{.*}} = bitcast <2 x i64> {{.*}} to <8 x i16>
  // OGCG: %{{.*}} = bitcast <8 x i16> %{{.*}} to <8 x half>
  // OGCG: %{{.*}} = fpext <8 x half> %{{.*}} to <8 x float>
  // OGCG: %{{.*}} = bitcast i8 {{.*}} to <8 x i1>
  // OGCG: %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> {{.*}}
  // OGCG: ret <8 x float> {{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps256_mask((__v8hi)a, _mm256_setzero_ps(), k);
}

__m512 test_vcvtph2ps512_maskz(__m256i a, __mmask16 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps512_maskz
  // CIR: %{{.*}} = cir.load {{.*}} : !cir.ptr<!cir.vector<4 x !s64i>>, !cir.vector<4 x !s64i>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<4 x !s64i> -> !cir.vector<16 x !s16i>
  // CIR: %{{.*}} = cir.call @_mm512_setzero_ps() : () -> !cir.vector<16 x !cir.float>
  // CIR: %{{.*}} = cir.load {{.*}} : !cir.ptr<!u16i>, !u16i
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s16i> -> !cir.vector<16 x !cir.f16>
  // CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.vector<16 x !cir.f16> -> !cir.vector<16 x !cir.float>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u16i -> !cir.vector<16 x !cir.{{(bool|int<s, 1>)}}>
  // CIR: cir.{{(select if|vec.ternary)}}{{.*}}%{{.*}}, %{{.*}}, %{{.*}}

  // LLVM-LABEL: @test_vcvtph2ps512_maskz
  // LLVM: %{{.*}} = bitcast <4 x i64> {{.*}} to <16 x i16>
  // LLVM: %{{.*}} = bitcast <16 x i16> %{{.*}} to <16 x half>
  // LLVM: %{{.*}} = fpext <16 x half> %{{.*}} to <16 x float>
  // LLVM: %{{.*}} = bitcast i16 {{.*}} to <16 x i1>
  // LLVM: %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> {{.*}}
  // LLVM: ret <16 x float> {{.*}}

  // OGCG-LABEL: @test_vcvtph2ps512_maskz
  // OGCG: %{{.*}} = bitcast <4 x i64> {{.*}} to <16 x i16>
  // OGCG: %{{.*}} = bitcast <16 x i16> %{{.*}} to <16 x half>
  // OGCG: %{{.*}} = fpext <16 x half> %{{.*}} to <16 x float>
  // OGCG: %{{.*}} = bitcast i16 {{.*}} to <16 x i1>
  // OGCG: %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> {{.*}}
  // OGCG: ret <16 x float> {{.*}}
  typedef short __v16hi __attribute__((__vector_size__(32)));
  return __builtin_ia32_vcvtph2ps512_mask((__v16hi)a, _mm512_setzero_ps(), k, 4);
}
