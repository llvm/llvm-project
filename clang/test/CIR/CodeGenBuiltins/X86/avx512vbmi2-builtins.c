// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding -triple x86_64-unknown-linux-gnu -fclangir -target-feature +avx512vbmi2 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding -triple x86_64-unknown-linux-gnu -fclangir -target-feature +avx512vbmi2 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vbmi2 -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s


#include <immintrin.h>

__m512i test_mm512_shldv_epi64(__m512i s, __m512i a, __m512i b) {
  // CIR-LABEL: @_mm512_shldv_epi64
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<8 x !u64i>
  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !u64i>, !cir.vector<8 x !u64i>, !cir.vector<8 x !u64i>) -> !cir.vector<8 x !u64i>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<8 x !u64i> -> !cir.vector<8 x !s64i>
  // CIR-LABEL: @test_mm512_shldv_epi64
  // CIR: %{{.*}} = cir.call @_mm512_shldv_epi64(%{{.*}}, %{{.*}}, %{{.*}}) : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !cir.vector<8 x !s64i>
  // LLVM-LABEL: @test_mm512_shldv_epi64
  // LLVM: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> {{.*}}, <8 x i64> {{.*}}, <8 x i64>
  // OGCG-LABEL: @test_mm512_shldv_epi64
  // OGCG: call <8 x i64> @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64>
  return _mm512_shldv_epi64(s, a, b);
}

