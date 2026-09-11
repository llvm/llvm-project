// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m128i test_mm_mul_epi32(__m128i x, __m128i y) {
  // CIR-LABEL: _mm_mul_epi32
  // CIR: [[A64:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !s64i>
  // CIR: [[B64:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !s64i>
  // CIR: [[SC:%.*]] = cir.const #cir.int<32> : !s64i
  // CIR: [[SV:%.*]] = cir.vec.splat [[SC]] : !s64i, !cir.vector<2 x !s64i>
  // CIR: [[SHL_A:%.*]]  = cir.shift(left, [[A64]] : !cir.vector<2 x !s64i>, [[SV]] : !cir.vector<2 x !s64i>)
  // CIR: [[ASHR_A:%.*]] = cir.shift(right, [[SHL_A]] : !cir.vector<2 x !s64i>, [[SV]] : !cir.vector<2 x !s64i>)
  // CIR: [[SHL_B:%.*]]  = cir.shift(left, [[B64]] : !cir.vector<2 x !s64i>, [[SV]] : !cir.vector<2 x !s64i>)
  // CIR: [[ASHR_B:%.*]] = cir.shift(right, [[SHL_B]] : !cir.vector<2 x !s64i>, [[SV]] : !cir.vector<2 x !s64i>)
  // CIR: [[MUL:%.*]]    = cir.mul [[ASHR_A]], [[ASHR_B]]

  // LLVM-LABEL: _mm_mul_epi32
  // LLVM: shl <2 x i64> %{{.*}}, splat (i64 32)
  // LLVM: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // LLVM: shl <2 x i64> %{{.*}}, splat (i64 32)
  // LLVM: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // LLVM: mul <2 x i64> %{{.*}}, %{{.*}}

  // OGCG-LABEL: _mm_mul_epi32
  // OGCG: shl <2 x i64> %{{.*}}, splat (i64 32)
  // OGCG: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // OGCG: shl <2 x i64> %{{.*}}, splat (i64 32)
  // OGCG: ashr <2 x i64> %{{.*}}, splat (i64 32)
  // OGCG: mul <2 x i64> %{{.*}}, %{{.*}}

  return _mm_mul_epi32(x, y);
}

__m128i test_mm_blend_epi16(__m128i V1, __m128i V2) {
  // CIR-LABEL: test_mm_blend_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<11> : !s32i, #cir.int<4> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !s16i>

  // LLVM-LABEL: test_mm_blend_epi16
  // LLVM: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>

  // OGCG-LABEL: test_mm_blend_epi16
  // OGCG: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>
  return _mm_blend_epi16(V1, V2, 42);
}

__m128d test_mm_blend_pd(__m128d V1, __m128d V2) {
  // CIR-LABEL: test_mm_blend_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_blend_pd
  // LLVM: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 3>

  // OGCG-LABEL: test_mm_blend_pd
  // OGCG: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 3>
  return _mm_blend_pd(V1, V2, 2);
}

__m128 test_mm_blend_ps(__m128 V1, __m128 V2) {
  // CIR-LABEL: test_mm_blend_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_blend_ps
  // LLVM: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 3>

  // OGCG-LABEL: test_mm_blend_ps
  // OGCG: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
  return _mm_blend_ps(V1, V2, 6);
}

__m128d test_mm_ceil_pd(__m128d x) {
  // CIR-LABEL: test_mm_ceil_pd
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.pd" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_ceil_pd
  // LLVM: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 2)

  // OGCG-LABEL: test_mm_ceil_pd
  // OGCG: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 2)
  return _mm_ceil_pd(x);
}

__m128 test_mm_ceil_ps(__m128 x) {
  // CIR-LABEL: test_mm_ceil_ps
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ps" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_ceil_ps
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 2)

  // OGCG-LABEL: test_mm_ceil_ps
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 2)
  return _mm_ceil_ps(x);
}

__m128d test_mm_ceil_sd(__m128d x, __m128d y) {
  // CIR-LABEL: test_mm_ceil_sd
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.sd" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_ceil_sd
  // LLVM: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 2)

  // OGCG-LABEL: test_mm_ceil_sd
  // OGCG: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 2)
  return _mm_ceil_sd(x, y);
}

__m128 test_mm_ceil_ss(__m128 x, __m128 y) {
  // CIR-LABEL: test_mm_ceil_ss
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ss" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_ceil_ss
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 2)

  // OGCG-LABEL: test_mm_ceil_ss
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 2)
  return _mm_ceil_ss(x, y);
}

__m128d test_mm_floor_pd(__m128d x) {
  // CIR-LABEL: test_mm_floor_pd
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.pd" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_floor_pd
  // LLVM: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 1)

  // OGCG-LABEL: test_mm_floor_pd
  // OGCG: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 1)
  return _mm_floor_pd(x);
}

__m128 test_mm_floor_ps(__m128 x) {
  // CIR-LABEL: test_mm_floor_ps
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ps" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_floor_ps
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 1)

  // OGCG-LABEL: test_mm_floor_ps
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 1)
  return _mm_floor_ps(x);
}

__m128d test_mm_floor_sd(__m128d x, __m128d y) {
  // CIR-LABEL: test_mm_floor_sd
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.sd" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_floor_sd
  // LLVM: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 1)

  // OGCG-LABEL: test_mm_floor_sd
  // OGCG: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 1)
  return _mm_floor_sd(x, y);
}

__m128 test_mm_floor_ss(__m128 x, __m128 y) {
  // CIR-LABEL: test_mm_floor_ss
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ss" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_floor_ss
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 1)

  // OGCG-LABEL: test_mm_floor_ss
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 1)
  return _mm_floor_ss(x, y);
}

__m128d test_mm_round_pd(__m128d x) {
  // CIR-LABEL: test_mm_round_pd
  // CIR: cir.roundeven %{{.*}} : !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_pd
  // LLVM: call <2 x double> @llvm.roundeven.v2f64(<2 x double> %{{.*}})

  // OGCG-LABEL: test_mm_round_pd
  // OGCG: call <2 x double> @llvm.roundeven.v2f64(<2 x double> %{{.*}})
  return _mm_round_pd(x, 0b1000);
}

__m128d test_mm_round_pd_mxcsr(__m128d x) {
  // CIR-LABEL: test_mm_round_pd_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.pd" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_pd_mxcsr
  // LLVM: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm_round_pd_mxcsr
  // OGCG: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 12)
  return _mm_round_pd(x, 0b1100);
}

__m128d test_mm_round_pd_fround_no_exc(__m128d x) {
  // CIR-LABEL: test_mm_round_pd_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.pd" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_pd_fround_no_exc
  // LLVM: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm_round_pd_fround_no_exc
  // OGCG: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 0)
  return _mm_round_pd(x, 0b0000);
}

__m128 test_mm_round_ps(__m128 x) {
  // CIR-LABEL: test_mm_round_ps
  // CIR: cir.floor %{{.*}} : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ps
  // LLVM: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{.*}})

  // OGCG-LABEL: test_mm_round_ps
  // OGCG: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{.*}})
  return _mm_round_ps(x, 0b1001);
}

__m128 test_mm_round_ps_mxcsr(__m128 x) {
  // CIR-LABEL: test_mm_round_ps_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ps" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ps_mxcsr
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm_round_ps_mxcsr
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 12)
  return _mm_round_ps(x, 0b1100);
}

__m128 test_mm_round_ps_fround_no_exc(__m128 x) {
  // CIR-LABEL: test_mm_round_ps_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ps" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ps_fround_no_exc
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm_round_ps_fround_no_exc
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 0)
  return _mm_round_ps(x, 0b0000);
}

__m128d test_mm_round_sd(__m128d x, __m128d y) {
  // CIR-LABEL: test_mm_round_sd
  // %[[A:.*]] = cir.vec.extract = %{{.*}}[%{{.*}} : !u64] : !cir.vector<2 x !cir.double>
  // %[[B:.*]] = cir.roundeven %[[A]] : !cir.double
  // cir.vec.insert = %[[B]], %{{.*}}[%{{.*}} : !u64] : !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_sd
  // LLVM: %[[A:.*]] = extractelement <2 x double> %{{.*}}, i64 0
  // LLVM: %[[B:.*]] = call double @llvm.roundeven.f64(double %[[A]])
  // LLVM: insertelement <2 x double> %{{.*}}, double %[[B]], i64 0

  // OGCG-LABEL: test_mm_round_sd
  // OGCG: %[[A:.*]] = extractelement <2 x double> %{{.*}}, i32 0
  // OGCG: %[[B:.*]] = call double @llvm.roundeven.f64(double %[[A]])
  // OGCG: insertelement <2 x double> %0, double %[[B]], i32 0
  return _mm_round_sd(x, y, 0b1000);
}

__m128d test_mm_round_sd_mxcsr(__m128d x, __m128d y) {
  // CIR-LABEL: test_mm_round_sd_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.sd" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_sd_mxcsr
  // LLVM: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm_round_sd_mxcsr
  // OGCG: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 12)
  return _mm_round_sd(x, y, 0b1100);
}


__m128d test_mm_round_sd_fround_no_exc(__m128d x, __m128d y) {
  // CIR-LABEL: test_mm_round_sd_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.sd" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_sd_fround_no_exc
  // LLVM: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm_round_sd_fround_no_exc
  // OGCG: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)
  return _mm_round_sd(x, y, 0b0000);
}

__m128 test_mm_round_ss(__m128 x, __m128 y) {
  // CIR-LABEL: test_mm_round_ss
  // %[[A:.*]] = cir.vec.extract = %{{.*}}[%{{.*}} : !u64] : !cir.vector<4 x !cir.float>
  // %[[B:.*]] = cir.trunc %[[A]] : !cir.float
  // cir.vec.insert = %[[B]], %{{.*}}[%{{.*}} : !u64] : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ss
  // LLVM: %[[A:.*]] = extractelement <4 x float> %{{.*}}, i64 0
  // LLVM: %[[B:.*]] = call float @llvm.trunc.f32(float %[[A]])
  // LLVM: insertelement <4 x float> %{{.*}}, float %[[B]], i64 0

  // OGCG-LABEL: test_mm_round_ss
  // OGCG: %[[A:.*]] = extractelement <4 x float> %{{.*}}, i32 0
  // OGCG: %[[B:.*]] = call float @llvm.trunc.f32(float %[[A]])
  // OGCG: insertelement <4 x float> %{{.*}}, float %[[B]], i32 0
  return _mm_round_ss(x, y, 0b1011);
}

__m128 test_mm_round_ss_mxcsr(__m128 x, __m128 y) {
  // CIR-LABEL: test_mm_round_ss_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ss" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ss_mxcsr
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm_round_ss_mxcsr
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 12)
  return _mm_round_ss(x, y, 0b1100);
}

__m128 test_mm_round_ss_fround_no_exc(__m128 x, __m128 y) {
  // CIR-LABEL: test_mm_round_ss_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ss" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ss_fround_no_exc
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm_round_ss_fround_no_exc
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  return _mm_round_ss(x, y, 0b0000);
}
