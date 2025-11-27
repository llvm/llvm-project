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

__m512i test_mm512_ror_epi32(__m512i __A) {
  // CIR-LABEL: test_mm512_ror_epi32
  // CIR: {{%.*}} =  cir.cast integral {{%.*}} : !s32i -> !u32i
  // CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u32i, !cir.vector<16 x !u32i>
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "fshr" {{%.*}}: (!cir.vector<16 x !s32i>, !cir.vector<16 x !s32i>, !cir.vector<16 x !u32i>) -> !cir.vector<16 x !s32i> 

  // LLVM-LABEL: test_mm512_ror_epi32
  // LLVM: %[[CASTED_VAR:.*]] = bitcast <8 x i64> {{%.*}} to <16 x i32>
  // LLVM: {{%.*}} = call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %[[CASTED_VAR]], <16 x i32> %[[CASTED_VAR]], <16 x i32> splat (i32 5))

  // OGCG-LABEL: test_mm512_ror_epi32
  // OGCG: %[[CASTED_VAR:.*]] = bitcast <8 x i64> {{%.*}} to <16 x i32>
  // OGCG: {{%.*}} = call <16 x i32> @llvm.fshr.v16i32(<16 x i32> %[[CASTED_VAR]], <16 x i32> %[[CASTED_VAR]], <16 x i32> splat (i32 5))
  return _mm512_ror_epi32(__A, 5); 
}

__m512i test_mm512_ror_epi64(__m512i __A) {
  // CIR-LABEL: test_mm512_ror_epi64
  // CIR: {{%.*}} =  cir.cast integral {{%.*}} : !s32i -> !u32i
  // CIR: {{%.*}} =  cir.cast integral {{%.*}} : !u32i -> !u64i
  // CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u64i, !cir.vector<8 x !u64i>
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "fshr" {{%.*}}: (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !u64i>) -> !cir.vector<8 x !s64i> 

  // LLVM-LABEL: test_mm512_ror_epi64
  // LLVM: %[[VAR:.*]] = load <8 x i64>, ptr {{%.*}}, align 64
  // LLVM: {{%.*}} = call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %[[VAR]], <8 x i64> %[[VAR]], <8 x i64> splat (i64 5))

  // OGCG-LABEL: test_mm512_ror_epi64
  // OGCG: %[[VAR:.*]] = load <8 x i64>, ptr {{%.*}}, align 64
  // OGCG: {{%.*}} = call <8 x i64> @llvm.fshr.v8i64(<8 x i64> %[[VAR]], <8 x i64> %[[VAR]], <8 x i64> splat (i64 5))
  return _mm512_ror_epi64(__A, 5); 
}
