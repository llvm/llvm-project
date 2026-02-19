// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll  -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll  -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vp2intersect -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

// CIR: !rec_anon_struct = !cir.record<struct  {!cir.vector<8 x !cir.bool>, !cir.vector<8 x !cir.bool>}>
// CIR: !rec_anon_struct1 = !cir.record<struct  {!cir.vector<4 x !cir.bool>, !cir.vector<4 x !cir.bool>}>
// CIR: !rec_anon_struct2 = !cir.record<struct  {!cir.vector<2 x !cir.bool>, !cir.vector<2 x !cir.bool>}>
void test_mm256_2intersect_epi32(__m256i a, __m256i b, __mmask8 *m0, __mmask8 *m1) {
  // CIR-LABEL: mm256_2intersect_epi32
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.vp2intersect.d.256" %{{.*}}, %{{.*}} : (!cir.vector<8 x !s32i>, !cir.vector<8 x !s32i>) -> !rec_anon_struct
  // CIR: %[[VAL1:.*]] = cir.extract_member %[[RES]][0] : !rec_anon_struct -> !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST1:.*]] = cir.cast bitcast %[[VAL1]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST1]], %{{.*}} : !u8i, !cir.ptr<!u8i>
  // CIR: %[[VAL2:.*]] = cir.extract_member %[[RES]][1] : !rec_anon_struct -> !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST2:.*]] = cir.cast bitcast %[[VAL2]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST2]], %{{.*}} : !u8i, !cir.ptr<!u8i>

  // LLVM-LABEL: test_mm256_2intersect_epi32
  // LLVM: %[[RES:.*]] = call { <8 x i1>, <8 x i1> } @llvm.x86.avx512.vp2intersect.d.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // LLVM: %[[VAL1:.*]] = extractvalue { <8 x i1>, <8 x i1> } %{{.*}}, 0
  // LLVM: %[[CAST1:.*]] = bitcast <8 x i1> %[[VAL1]] to i8
  // LLVM: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // LLVM: %[[VAL2:.*]] = extractvalue { <8 x i1>, <8 x i1> } %{{.*}}, 1
  // LLVM: %[[CAST2:.*]] = bitcast <8 x i1> %[[VAL2]] to i8
  // LLVM: store i8 %[[CAST2]], ptr %{{.*}}, align 1

  // OGCG-LABEL: test_mm256_2intersect_epi32
  // OGCG: %[[RES:.*]] = call { <8 x i1>, <8 x i1> } @llvm.x86.avx512.vp2intersect.d.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // OGCG: %[[VAL1:.*]] = extractvalue { <8 x i1>, <8 x i1> } %{{.*}}, 0
  // OGCG: %[[CAST1:.*]] = bitcast <8 x i1> %[[VAL1]] to i8
  // OGCG: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // OGCG: %[[VAL2:.*]] = extractvalue { <8 x i1>, <8 x i1> } %{{.*}}, 1
  // OGCG: %[[CAST2:.*]] = bitcast <8 x i1> %[[VAL2]] to i8
  // OGCG: store i8 %[[CAST2]], ptr %{{.*}}, align 1
  _mm256_2intersect_epi32(a, b, m0, m1);
}

void test_mm256_2intersect_epi64(__m256i a, __m256i b, __mmask8 *m0, __mmask8 *m1) {
  // CIR-LABEL: mm256_2intersect_epi64
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.vp2intersect.q.256" %{{.*}}, %{{.*}} : (!cir.vector<4 x !s64i>, !cir.vector<4 x !s64i>) -> !rec_anon_struct1
  // CIR: %[[VAL1:.*]] = cir.extract_member %[[RES]][0] : !rec_anon_struct1 -> !cir.vector<4 x !cir.bool>
  // CIR: %[[ZERO1:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.bool>
  // CIR: %[[SHUF1:.*]] = cir.vec.shuffle(%[[VAL1]], %[[ZERO1]] : !cir.vector<4 x !cir.bool>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST1:.*]] = cir.cast bitcast %[[SHUF1]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST1]], %{{.*}} : !u8i, !cir.ptr<!u8i>
  // CIR: %[[VAL2:.*]] = cir.extract_member %[[RES]][1] : !rec_anon_struct1 -> !cir.vector<4 x !cir.bool>
  // CIR: %[[ZERO2:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.bool>
  // CIR: %[[SHUF2:.*]] = cir.vec.shuffle(%[[VAL2]], %[[ZERO2]] : !cir.vector<4 x !cir.bool>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST2:.*]] = cir.cast bitcast %[[SHUF2]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST2]], %{{.*}} : !u8i, !cir.ptr<!u8i>

  // LLVM-LABEL: test_mm256_2intersect_epi64
  // LLVM: %[[RES:.*]] = call { <4 x i1>, <4 x i1> } @llvm.x86.avx512.vp2intersect.q.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // LLVM: %[[VAL1:.*]] = extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 0
  // LLVM: %[[SHUF1:.*]] = shufflevector <4 x i1> %[[VAL1]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: %[[CAST1:.*]] = bitcast <8 x i1> %[[SHUF1]] to i8
  // LLVM: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // LLVM: %[[VAL2:.*]] = extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 1
  // LLVM: %[[SHUF2:.*]] = shufflevector <4 x i1> %[[VAL2]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: %[[CAST2:.*]] = bitcast <8 x i1> %[[SHUF2]] to i8
  // LLVM: store i8 %[[CAST2]], ptr %{{.*}}, align 1

  // OGCG-LABEL: test_mm256_2intersect_epi64
  // OGCG: %[[RES:.*]] = call { <4 x i1>, <4 x i1> } @llvm.x86.avx512.vp2intersect.q.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // OGCG: %[[VAL1:.*]] = extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 0
  // OGCG: %[[SHUF1:.*]] = shufflevector <4 x i1> %[[VAL1]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: %[[CAST1:.*]] = bitcast <8 x i1> %[[SHUF1]] to i8
  // OGCG: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // OGCG: %[[VAL2:.*]] = extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 1
  // OGCG: %[[SHUF2:.*]] = shufflevector <4 x i1> %[[VAL2]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: %[[CAST2:.*]] = bitcast <8 x i1> %[[SHUF2]] to i8
  // OGCG: store i8 %[[CAST2]], ptr %{{.*}}, align 1
  _mm256_2intersect_epi64(a, b, m0, m1);
}

void test_mm_2intersect_epi32(__m128i a, __m128i b, __mmask8 *m0, __mmask8 *m1) {
  // CIR-LABEL: mm_2intersect_epi32
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.vp2intersect.d.128" %{{.*}}, %{{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>) -> !rec_anon_struct1
  // CIR: %[[VAL1:.*]] = cir.extract_member %[[RES]][0] : !rec_anon_struct1 -> !cir.vector<4 x !cir.bool>
  // CIR: %[[ZERO1:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.bool>
  // CIR: %[[SHUF1:.*]] = cir.vec.shuffle(%[[VAL1]], %[[ZERO1]] : !cir.vector<4 x !cir.bool>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST1:.*]] = cir.cast bitcast %[[SHUF1]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST1]], %{{.*}} : !u8i, !cir.ptr<!u8i>
  // CIR: %[[VAL2:.*]] = cir.extract_member %[[RES]][1] : !rec_anon_struct1 -> !cir.vector<4 x !cir.bool>
  // CIR: %[[ZERO2:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.bool>
  // CIR: %[[SHUF2:.*]] = cir.vec.shuffle(%[[VAL2]], %[[ZERO2]] : !cir.vector<4 x !cir.bool>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<4> : !s64i, #cir.int<5> : !s64i, #cir.int<6> : !s64i, #cir.int<7> : !s64i] : !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST2:.*]] = cir.cast bitcast %[[SHUF2]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST2]], %{{.*}} : !u8i, !cir.ptr<!u8i>

  // LLVM-LABEL: test_mm_2intersect_epi32
  // LLVM: %[[RES:.*]] = call { <4 x i1>, <4 x i1> } @llvm.x86.avx512.vp2intersect.d.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // LLVM: %[[VAL1:.*]] = extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 0
  // LLVM: %[[SHUF1:.*]] = shufflevector <4 x i1> %[[VAL1]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: %[[CAST1:.*]] = bitcast <8 x i1> %[[SHUF1]] to i8
  // LLVM: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // LLVM: %[[VAL2:.*]] = extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 1
  // LLVM: %[[SHUF2:.*]] = shufflevector <4 x i1> %[[VAL2]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: %[[CAST2:.*]] = bitcast <8 x i1> %[[SHUF2]] to i8
  // LLVM: store i8 %[[CAST2]], ptr %{{.*}}, align 1

  // OGCG-LABEL: test_mm_2intersect_epi32
  // OGCG: %[[RES:.*]] = call { <4 x i1>, <4 x i1> } @llvm.x86.avx512.vp2intersect.d.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // OGCG: %[[VAL1:.*]] = extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 0
  // OGCG: %[[SHUF1:.*]] = shufflevector <4 x i1> %[[VAL1]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: %[[CAST1:.*]] = bitcast <8 x i1> %[[SHUF1]] to i8
  // OGCG: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // OGCG: %[[VAL2:.*]] = extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 1
  // OGCG: %[[SHUF2:.*]] = shufflevector <4 x i1> %[[VAL2]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG: %[[CAST2:.*]] = bitcast <8 x i1> %[[SHUF2]] to i8
  // OGCG: store i8 %[[CAST2]], ptr %{{.*}}, align 1
  _mm_2intersect_epi32(a, b, m0, m1);
}

void test_mm_2intersect_epi64(__m128i a, __m128i b, __mmask8 *m0, __mmask8 *m1) {
  // CIR-LABEL: mm_2intersect_epi64
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.vp2intersect.q.128" %{{.*}}, %{{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>) -> !rec_anon_struct2
  // CIR: %[[VAL1:.*]] = cir.extract_member %[[RES]][0] : !rec_anon_struct2 -> !cir.vector<2 x !cir.bool>
  // CIR: %[[ZERO1:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.bool>
  // CIR: %[[SHUF1:.*]] = cir.vec.shuffle(%[[VAL1]], %[[ZERO1]] : !cir.vector<2 x !cir.bool>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i] : !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST1:.*]] = cir.cast bitcast %[[SHUF1]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST1]], %{{.*}} : !u8i, !cir.ptr<!u8i>
  // CIR: %[[VAL2:.*]] = cir.extract_member %[[RES]][1] : !rec_anon_struct2 -> !cir.vector<2 x !cir.bool>
  // CIR: %[[ZERO2:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.bool>
  // CIR: %[[SHUF2:.*]] = cir.vec.shuffle(%[[VAL2]], %[[ZERO2]] : !cir.vector<2 x !cir.bool>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i, #cir.int<2> : !s64i, #cir.int<3> : !s64i] : !cir.vector<8 x !cir.bool>
  // CIR: %[[CAST2:.*]] = cir.cast bitcast %[[SHUF2]] : !cir.vector<8 x !cir.bool> -> !u8i
  // CIR: cir.store align(1) %[[CAST2]], %{{.*}} : !u8i, !cir.ptr<!u8i>

  // LLVM-LABEL: test_mm_2intersect_epi64
  // LLVM: %[[RES:.*]] = call { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // LLVM: %[[VAL1:.*]] = extractvalue { <2 x i1>, <2 x i1> } %{{.*}}, 0
  // LLVM: %[[SHUF1:.*]] = shufflevector <2 x i1> %[[VAL1]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // LLVM: %[[CAST1:.*]] = bitcast <8 x i1> %[[SHUF1]] to i8
  // LLVM: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // LLVM: %[[VAL2:.*]] = extractvalue { <2 x i1>, <2 x i1> } %{{.*}}, 1
  // LLVM: %[[SHUF2:.*]] = shufflevector <2 x i1> %[[VAL2]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // LLVM: %[[CAST2:.*]] = bitcast <8 x i1> %[[SHUF2]] to i8
  // LLVM: store i8 %[[CAST2]], ptr %{{.*}}, align 1

  // OGCG-LABEL: test_mm_2intersect_epi64
  // OGCG: %[[RES:.*]] = call { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // OGCG: %[[VAL1:.*]] = extractvalue { <2 x i1>, <2 x i1> } %{{.*}}, 0
  // OGCG: %[[SHUF1:.*]] = shufflevector <2 x i1> %[[VAL1]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // OGCG: %[[CAST1:.*]] = bitcast <8 x i1> %[[SHUF1]] to i8
  // OGCG: store i8 %[[CAST1]], ptr %{{.*}}, align 1
  // OGCG: %[[VAL2:.*]] = extractvalue { <2 x i1>, <2 x i1> } %{{.*}}, 1
  // OGCG: %[[SHUF2:.*]] = shufflevector <2 x i1> %[[VAL2]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  // OGCG: %[[CAST2:.*]] = bitcast <8 x i1> %[[SHUF2]] to i8
  // OGCG: store i8 %[[CAST2]], ptr %{{.*}}, align 1
  _mm_2intersect_epi64(a, b, m0, m1);
}
