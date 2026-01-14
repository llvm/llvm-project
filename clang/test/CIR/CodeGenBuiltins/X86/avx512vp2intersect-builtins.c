// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -fclangir -emit-llvm -o %t.ll  -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -fclangir -emit-llvm -o %t.ll  -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>


// CIR: !rec_anon_struct = !cir.record<struct  {!cir.vector<16 x !cir.bool>, !cir.vector<16 x !cir.bool>}>
// CIR: !rec_anon_struct1 = !cir.record<struct  {!cir.vector<8 x !cir.bool>, !cir.vector<8 x !cir.bool>}>
void test_mm512_2intersect_epi32(__m512i a, __m512i b, __mmask16 *m0, __mmask16 *m1) {
  // CIR-LABEL: mm512_2intersect_epi32
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.vp2intersect.d.512" %{{.*}}, %{{.*}} : (!cir.vector<16 x !s32i>, !cir.vector<16 x !s32i>) -> !rec_anon_struct
  // CIR: %[[VAL1:.*]] = cir.extract_member %[[RES]][0] : !rec_anon_struct -> !cir.vector<16 x !cir.bool>
  // CIR: %[[CAST1:.*]] = cir.cast bitcast %[[VAL1]] : !cir.vector<16 x !cir.bool> -> !u16i
  // CIR: cir.store align(2) %[[CAST1]], %{{.*}} : !u16i, !cir.ptr<!u16i>
  // CIR: %[[VAL2:.*]] = cir.extract_member %[[RES]][1] : !rec_anon_struct -> !cir.vector<16 x !cir.bool>
  // CIR: %[[CAST2:.*]] = cir.cast bitcast %[[VAL2]] : !cir.vector<16 x !cir.bool> -> !u16i
  // CIR: cir.store align(2) %[[CAST2]], %{{.*}} : !u16i, !cir.ptr<!u16i>

  // LLVM-LABEL: test_mm512_2intersect_epi32
  // LLVM: %[[RES:.*]] = call { <16 x i1>, <16 x i1> } @llvm.x86.avx512.vp2intersect.d.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // LLVM: %[[VAL1:.*]] = extractvalue { <16 x i1>, <16 x i1> } %[[RES]], 0
  // LLVM: %[[CAST1:.*]] = bitcast <16 x i1> %[[VAL1]] to i16
  // LLVM: store i16 %[[CAST1]], ptr %{{.*}}, align 2
  // LLVM: %[[VAL2:.*]] = extractvalue { <16 x i1>, <16 x i1> } %[[RES]], 1
  // LLVM: %[[CAST2:.*]] = bitcast <16 x i1> %[[VAL2]] to i16
  // LLVM: store i16 %[[CAST2]], ptr %{{.*}}, align 2

  // OGCG-LABEL: test_mm512_2intersect_epi32
  // OGCG: %[[RES:.*]] = call { <16 x i1>, <16 x i1> } @llvm.x86.avx512.vp2intersect.d.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // OGCG: %[[VAL1:.*]] = extractvalue { <16 x i1>, <16 x i1> } %[[RES]], 0
  // OGCG: %[[CAST1:.*]] = bitcast <16 x i1> %[[VAL1]] to i16
  // OGCG: store i16 %[[CAST1]], ptr %{{.*}}, align 2
  // OGCG: %[[VAL2:.*]] = extractvalue { <16 x i1>, <16 x i1> } %[[RES]], 1
  // OGCG: %[[CAST2:.*]] = bitcast <16 x i1> %[[VAL2]] to i16
  // OGCG: store i16 %[[CAST2]], ptr %{{.*}}, align 2
  _mm512_2intersect_epi32(a, b, m0, m1);
}

void test_mm512_2intersect_epi64(__m512i a, __m512i b, __mmask8 *m0, __mmask8 *m1) {
  // CIR-LABEL: mm512_2intersect_epi64
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.vp2intersect.q.512" %{{.*}}, %{{.*}} : (!cir.vector<8 x !s64i>, !cir.vector<8 x !s64i>) -> !rec_anon_struct1
  // CIR: %[[VAL1:.*]] = cir.extract_member %[[RES]][0] : !rec_anon_struct1 -> !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST1:.*]] = cir.cast bitcast %[[VAL1]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST1]], %{{.*}} : !u8i, !cir.ptr<!u8i>
  // CIR: %[[VAL2:.*]] = cir.extract_member %[[RES]][1] : !rec_anon_struct1 -> !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST2:.*]] = cir.cast bitcast %[[VAL2]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST2]], %{{.*}} : !u8i, !cir.ptr<!u8i>

  // LLVM-LABEL: test_mm512_2intersect_epi64
  // LLVM: %[[RES:.*]] = call { <8 x i1>, <8 x i1> } @llvm.x86.avx512.vp2intersect.q.512(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // LLVM: %[[VAL1:.*]] = extractvalue { <8 x i1>, <8 x i1> } %[[RES]], 0
  // LLVM: %[[CAST1:.*]] = bitcast <8 x i1> %[[VAL1]] to i8
  // LLVM: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // LLVM: %[[VAL2:.*]] = extractvalue { <8 x i1>, <8 x i1> } %[[RES]], 1
  // LLVM: %[[CAST2:.*]] = bitcast <8 x i1> %[[VAL2]] to i8
  // LLVM: store i8 %[[CAST2]], ptr %{{.*}}, align 1

  // OGCG-LABEL: test_mm512_2intersect_epi64
  // OGCG: %[[RES:.*]] = call { <8 x i1>, <8 x i1> } @llvm.x86.avx512.vp2intersect.q.512(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // OGCG: %[[VAL1:.*]] = extractvalue { <8 x i1>, <8 x i1> } %[[RES]], 0
  // OGCG: %[[CAST1:.*]] = bitcast <8 x i1> %[[VAL1]] to i8
  // OGCG: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // OGCG: %[[VAL2:.*]] = extractvalue { <8 x i1>, <8 x i1> } %[[RES]], 1
  // OGCG: %[[CAST2:.*]] = bitcast <8 x i1> %[[VAL2]] to i8
  // OGCG: store i8 %[[CAST2]], ptr %{{.*}}, align 1
  _mm512_2intersect_epi64(a, b, m0, m1);
}
