// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -emit-llvm -o - -Wall -Werror | FileCheck --check-prefixes=COMMON,UNCONSTRAINED %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -ffp-exception-behavior=strict -emit-llvm -o - -Wall -Werror | FileCheck --check-prefixes=COMMON,STRICT %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -S -o - -Wall -Werror | FileCheck --check-prefix=CHECK-ASM %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -ffp-exception-behavior=strict -S -o - -Wall -Werror | FileCheck --check-prefix=CHECK-ASM %s
//
// At -O2 the default (non-strictfp) intrinsic is folded to a plain fadd, while
// the strictfp form is preserved (see test_mm512_add_round_ps_fold below).
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -O2 -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=FOLD %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -ffp-exception-behavior=strict -O2 -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=KEEP %s

// The packed add/sub/mul/div "_round" builtins with _MM_FROUND_CUR_DIRECTION
// lower to the unmasked x86 intrinsic with rounding operand 4.
//
// Without -ffp-exception-behavior=strict the call is a plain (non-strictfp)
// intrinsic call: under the default FP environment the optimizer is free to
// fold it to round-to-nearest IR.
//
// With -ffp-exception-behavior=strict the enclosing function and the call are
// marked "strictfp". That attribute is what makes the rest of the pipeline
// (InstCombine and the X86 SelectionDAG lowering) preserve the operation so it
// honors the live MXCSR rounding mode instead of constant-folding it.

#include <immintrin.h>

__m512 test_mm512_add_round_ps(__m512 a, __m512 b) {
  // COMMON-LABEL: test_mm512_add_round_ps
  // UNCONSTRAINED: call <16 x float> @llvm.x86.avx512.add.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  // STRICT: call <16 x float> @llvm.x86.avx512.add.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4) #[[ATTR:[0-9]+]]
  // CHECK-ASM: vaddps
  return _mm512_add_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512 test_mm512_sub_round_ps(__m512 a, __m512 b) {
  // COMMON-LABEL: test_mm512_sub_round_ps
  // UNCONSTRAINED: call <16 x float> @llvm.x86.avx512.sub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  // STRICT: call <16 x float> @llvm.x86.avx512.sub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4) #[[ATTR]]
  // CHECK-ASM: vsubps
  return _mm512_sub_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512 test_mm512_mul_round_ps(__m512 a, __m512 b) {
  // COMMON-LABEL: test_mm512_mul_round_ps
  // UNCONSTRAINED: call <16 x float> @llvm.x86.avx512.mul.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  // STRICT: call <16 x float> @llvm.x86.avx512.mul.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4) #[[ATTR]]
  // CHECK-ASM: vmulps
  return _mm512_mul_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512 test_mm512_div_round_ps(__m512 a, __m512 b) {
  // COMMON-LABEL: test_mm512_div_round_ps
  // UNCONSTRAINED: call <16 x float> @llvm.x86.avx512.div.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  // STRICT: call <16 x float> @llvm.x86.avx512.div.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4) #[[ATTR]]
  // CHECK-ASM: vdivps
  return _mm512_div_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
}

// Optimized (-O2) view of the same operation, equivalent to the InstCombine
// test add_ps_512_cur_direction: without strictfp the rounding-mode operand is
// dropped and the call becomes a plain fadd that no longer carries any MXCSR
// dependence; with strictfp the intrinsic (and its MXCSR dependence) survives.
__m512 test_mm512_add_round_ps_fold(__m512 a, __m512 b) {
  // FOLD-LABEL: @test_mm512_add_round_ps_fold(
  // FOLD: fadd <16 x float> %{{.*}}, %{{.*}}
  // KEEP-LABEL: @test_mm512_add_round_ps_fold(
  // KEEP: call <16 x float> @llvm.x86.avx512.add.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  return _mm512_add_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512 test_mm512_sub_round_ps_fold(__m512 a, __m512 b) {
  // FOLD-LABEL: @test_mm512_sub_round_ps_fold(
  // FOLD: fsub <16 x float> %{{.*}}, %{{.*}}
  // KEEP-LABEL: @test_mm512_sub_round_ps_fold(
  // KEEP: call <16 x float> @llvm.x86.avx512.sub.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  return _mm512_sub_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512 test_mm512_mul_round_ps_fold(__m512 a, __m512 b) {
  // FOLD-LABEL: @test_mm512_mul_round_ps_fold(
  // FOLD: fmul <16 x float> %{{.*}}, %{{.*}}
  // KEEP-LABEL: @test_mm512_mul_round_ps_fold(
  // KEEP: call <16 x float> @llvm.x86.avx512.mul.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  return _mm512_mul_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512 test_mm512_div_round_ps_fold(__m512 a, __m512 b) {
  // FOLD-LABEL: @test_mm512_div_round_ps_fold(
  // FOLD: fdiv <16 x float> %{{.*}}, %{{.*}}
  // KEEP-LABEL: @test_mm512_div_round_ps_fold(
  // KEEP: call <16 x float> @llvm.x86.avx512.div.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 4)
  return _mm512_div_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512d test_mm512_add_round_pd_fold(__m512d a, __m512d b) {
  // FOLD-LABEL: @test_mm512_add_round_pd_fold(
  // FOLD: fadd <8 x double> %{{.*}}, %{{.*}}
  // KEEP-LABEL: @test_mm512_add_round_pd_fold(
  // KEEP: call <8 x double> @llvm.x86.avx512.add.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4)
  return _mm512_add_round_pd(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512d test_mm512_sub_round_pd_fold(__m512d a, __m512d b) {
  // FOLD-LABEL: @test_mm512_sub_round_pd_fold(
  // FOLD: fsub <8 x double> %{{.*}}, %{{.*}}
  // KEEP-LABEL: @test_mm512_sub_round_pd_fold(
  // KEEP: call <8 x double> @llvm.x86.avx512.sub.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4)
  return _mm512_sub_round_pd(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512d test_mm512_mul_round_pd_fold(__m512d a, __m512d b) {
  // FOLD-LABEL: @test_mm512_mul_round_pd_fold(
  // FOLD: fmul <8 x double> %{{.*}}, %{{.*}}
  // KEEP-LABEL: @test_mm512_mul_round_pd_fold(
  // KEEP: call <8 x double> @llvm.x86.avx512.mul.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4)
  return _mm512_mul_round_pd(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512d test_mm512_div_round_pd_fold(__m512d a, __m512d b) {
  // FOLD-LABEL: @test_mm512_div_round_pd_fold(
  // FOLD: fdiv <8 x double> %{{.*}}, %{{.*}}
  // KEEP-LABEL: @test_mm512_div_round_pd_fold(
  // KEEP: call <8 x double> @llvm.x86.avx512.div.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4)
  return _mm512_div_round_pd(a, b, _MM_FROUND_CUR_DIRECTION);
}

__m512d test_mm512_add_round_pd(__m512d a, __m512d b) {
  // COMMON-LABEL: test_mm512_add_round_pd
  // UNCONSTRAINED: call <8 x double> @llvm.x86.avx512.add.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4)
  // STRICT: call <8 x double> @llvm.x86.avx512.add.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 4) #[[ATTR]]
  // CHECK-ASM: vaddpd
  return _mm512_add_round_pd(a, b, _MM_FROUND_CUR_DIRECTION);
}

// STRICT: attributes #[[ATTR]] = { strictfp }
