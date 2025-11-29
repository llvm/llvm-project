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
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<16 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<16 x !cir.float>

  // LLVM-LABEL: test_mm512_undefined
  // LLVM: store <16 x float> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <16 x float>, ptr %[[A]], align 64
  // LLVM: ret <16 x float> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined
  // OGCG: ret <16 x float> zeroinitializer
  return _mm512_undefined();
}

__m512 test_mm512_undefined_ps(void) {
  // CIR-LABEL: _mm512_undefined_ps
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<16 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<16 x !cir.float>

  // LLVM-LABEL: test_mm512_undefined_ps
  // LLVM: store <16 x float> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <16 x float>, ptr %[[A]], align 64
  // LLVM: ret <16 x float> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_ps
  // OGCG: ret <16 x float> zeroinitializer
  return _mm512_undefined_ps();
}

__m512d test_mm512_undefined_pd(void) {
  // CIR-LABEL: _mm512_undefined_pd
  // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.double>

  // LLVM-LABEL: test_mm512_undefined_pd
  // LLVM: store <8 x double> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <8 x double>, ptr %[[A]], align 64
  // LLVM: ret <8 x double> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_pd
  // OGCG: ret <8 x double> zeroinitializer
  return _mm512_undefined_pd();
}

__m512i test_mm512_undefined_epi32(void) {
  // CIR-LABEL: _mm512_undefined_epi32
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<8 x !s64i>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !s64i>

  // LLVM-LABEL: test_mm512_undefined_epi32
  // LLVM: store <8 x i64> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <8 x i64>, ptr %[[A]], align 64
  // LLVM: ret <8 x i64> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_epi32
  // OGCG: ret <8 x i64> zeroinitializer
  return _mm512_undefined_epi32();
}

__m256 test_mm512_i64gather_ps(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i64gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qps.512"

  // LLVM-LABEL: test_mm512_i64gather_ps
  // LLVM: call <8 x float> @llvm.x86.avx512.mask.gather.qps.512

  // OGCG-LABEL: test_mm512_i64gather_ps
  // OGCG: call <8 x float> @llvm.x86.avx512.mask.gather.qps.512
  return _mm512_i64gather_ps(__index, __addr, 2);
}

__m256 test_mm512_mask_i64gather_ps(__m256 __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i64gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qps.512"

  // LLVM-LABEL: test_mm512_mask_i64gather_ps
  // LLVM: call <8 x float> @llvm.x86.avx512.mask.gather.qps.512

  // OGCG-LABEL: test_mm512_mask_i64gather_ps
  // OGCG: call <8 x float> @llvm.x86.avx512.mask.gather.qps.512
  return _mm512_mask_i64gather_ps(__v1_old, __mask, __index, __addr, 2);
}

__m256i test_mm512_i64gather_epi32(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i64gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpi.512"

  // LLVM-LABEL: test_mm512_i64gather_epi32
  // LLVM: call <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512

  // OGCG-LABEL: test_mm512_i64gather_epi32
  // OGCG: call <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512
  return _mm512_i64gather_epi32(__index, __addr, 2);
}

__m256i test_mm512_mask_i64gather_epi32(__m256i __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i64gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpi.512"

  // LLVM-LABEL: test_mm512_mask_i64gather_epi32
  // LLVM: call <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512

  // OGCG-LABEL: test_mm512_mask_i64gather_epi32
  // OGCG: call <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512
  return _mm512_mask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2);
}

__m512d test_mm512_i64gather_pd(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i64gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpd.512

  // LLVM-LABEL: test_mm512_i64gather_pd
  // LLVM: call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512

  // OGCG-LABEL: test_mm512_i64gather_pd
  // OGCG: call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512
  return _mm512_i64gather_pd(__index, __addr, 2);
}

__m512d test_mm512_mask_i64gather_pd(__m512d __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i64gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpd.512

  // LLVM-LABEL: test_mm512_mask_i64gather_pd
  // LLVM: call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512

  // OGCG-LABEL: test_mm512_mask_i64gather_pd
  // OGCG: call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512
  return _mm512_mask_i64gather_pd(__v1_old, __mask, __index, __addr, 2);
}

__m512i test_mm512_i64gather_epi64(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i64gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpq.512

  // LLVM-LABEL: test_mm512_i64gather_epi64
  // LLVM: call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512

  // OGCG-LABEL: test_mm512_i64gather_epi64
  // OGCG: call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512
  return _mm512_i64gather_epi64(__index, __addr, 2);
}

__m512i test_mm512_mask_i64gather_epi64(__m512i __v1_old, __mmask8 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i64gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.qpq.512

  // LLVM-LABEL: test_mm512_mask_i64gather_epi64
  // LLVM: call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512

  // OGCG-LABEL: test_mm512_mask_i64gather_epi64
  // OGCG: call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512
  return _mm512_mask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2);
}

__m512 test_mm512_i32gather_ps(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i32gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dps.512

  // LLVM-LABEL: test_mm512_i32gather_ps
  // LLVM: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512

  // OGCG-LABEL: test_mm512_i32gather_ps
  // OGCG: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512
  return _mm512_i32gather_ps(__index, __addr, 2);
}

__m512 test_mm512_mask_i32gather_ps(__m512 v1_old, __mmask16 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i32gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dps.512

  // LLVM-LABEL: test_mm512_mask_i32gather_ps
  // LLVM: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512

  // OGCG-LABEL: test_mm512_mask_i32gather_ps
  // OGCG: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512
  return _mm512_mask_i32gather_ps(v1_old, __mask, __index, __addr, 2);
}

__m512i test_mm512_i32gather_epi32(__m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i32gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpi.512

  // LLVM-LABEL: test_mm512_i32gather_epi32
  // LLVM: call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512

  // OGCG-LABEL: test_mm512_i32gather_epi32
  // OGCG: call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512
  return _mm512_i32gather_epi32(__index, __addr, 2);
}

__m512i test_mm512_mask_i32gather_epi32(__m512i __v1_old, __mmask16 __mask, __m512i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i32gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpi.512

  // LLVM-LABEL: test_mm512_mask_i32gather_epi32
  // LLVM: call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512

  // OGCG-LABEL: test_mm512_mask_i32gather_epi32
  // OGCG: call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512
  return _mm512_mask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2);
}

__m512d test_mm512_i32gather_pd(__m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i32gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpd.512

  // LLVM-LABEL: test_mm512_i32gather_pd
  // LLVM: call <8 x double> @llvm.x86.avx512.mask.gather.dpd.512

  // OGCG-LABEL: test_mm512_i32gather_pd
  // OGCG: call <8 x double> @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_i32gather_pd(__index, __addr, 2);
}

__m512d test_mm512_mask_i32gather_pd(__m512d __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i32gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpd.512

  // LLVM-LABEL: test_mm512_mask_i32gather_pd
  // LLVM: call <8 x double> @llvm.x86.avx512.mask.gather.dpd.512

  // OGCG-LABEL: test_mm512_mask_i32gather_pd
  // OGCG: call <8 x double> @llvm.x86.avx512.mask.gather.dpd.512
  return _mm512_mask_i32gather_pd(__v1_old, __mask, __index, __addr, 2);
}

__m512i test_mm512_i32gather_epi64(__m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_i32gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpq.512

  // LLVM-LABEL: test_mm512_i32gather_epi64
  // LLVM: call <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512
 
  // OGCG-LABEL: test_mm512_i32gather_epi64
  // OGCG: call <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_i32gather_epi64(__index, __addr, 2);
}

__m512i test_mm512_mask_i32gather_epi64(__m512i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm512_mask_i32gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather.dpq.512

  // LLVM-LABEL: test_mm512_mask_i32gather_epi64
  // LLVM: call <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512
 
  // OGCG-LABEL: test_mm512_mask_i32gather_epi64
  // OGCG: call <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512
  return _mm512_mask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2);
}
